//
// Copyright 2010-2011,2014 Ettus Research LLC
// Copyright 2018 Ettus Research, a National Instruments Company
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#include <uhd/types/tune_request.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <fstream>
#include <csignal>
#include <complex>
#include <thread>
#include <chrono>

// includes, project
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

namespace po = boost::program_options;

static bool stop_signal_called = false;
void sig_int_handler(int){stop_signal_called = true;}

#define num_buffers 384

typedef std::complex<float> sample_t;

typedef struct mdbuff
{
    std::vector<sample_t> data;
    uhd::rx_metadata_t    md;
} mdbuff_t;

__global__ void calcPow(cuFloatComplex *a, int N)
{
    int n   = N/blockDim.x;
    int idx = threadIdx.x*n;
    for (int i = idx; i < idx+n; i++) {
        a[i] = cuCmulf(a[i], cuConjf(a[i]));
    }
}

__global__ void calcSum(float *d, float *a, int N)
{
    int n   = N/blockDim.x;
    int idx = threadIdx.x*n;
    for (int i = idx; i < idx+n; i++) {
        a[i] += d[i*2];
    }
}

__global__ void calcMax(float *d, float *x, int N)
{
    int n   = N/blockDim.x;
    int idx = threadIdx.x*n;
    for (int i = idx; i < idx+n; i++) {
        x[i] = fmaxf(x[i], d[i*2]);
    }
}

__global__ void normalize(float *d, int N, float factor)
{
    int n   = N/blockDim.x;
    int idx = threadIdx.x*n;
    for (int i = idx; i < idx+n; i++) {
        d[i] /= factor;
    }
}

void cuda_handler(boost::circular_buffer<mdbuff_t *> *cb, uint32_t bufsz, uint32_t accum)
{
    cuFloatComplex *d_i; 
    float          *d_a; 
    float          *d_x; 
    cufftHandle plan;
    mdbuff_t *buf;

    float *h_a;
    float *h_x;

    uint32_t cnt = 0;

    FILE *ofile = fopen("test.dat", "wb"); //remove

    // Allocate output buffers
    h_a = (float *)malloc(sizeof(float) * bufsz);
    h_x = (float *)malloc(sizeof(float) * bufsz);

    // Allocate the input buffer
    cudaMalloc((void **)(&d_i), sizeof(cuFloatComplex) * bufsz);

    // Allocate the accumulator
    cudaMalloc((void **)(&d_a), sizeof(float) * bufsz);
    cudaMemset(d_a, 0, sizeof(float) * bufsz);

    // Allocate the peak detector
    cudaMalloc((void **)(&d_x), sizeof(float) * bufsz);
    cudaMemset(d_x, 0, sizeof(float) * bufsz);

    // Generate the FFT plan.
    cufftPlan1d(&plan, bufsz, CUFFT_C2C, 1);

    // Loop till stopped.
    while (!stop_signal_called) 
    {
        // Avoid tight-looping the processor. 
        while (cb->empty())
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        }

        // Copy the data up to the GPU and pop it off the circular buffer
        buf = (*cb)[0];
        std::cout << buf->md.to_pp_string() << std::endl;
        cudaMemcpy(d_i, &buf->data.front(), sizeof(cuFloatComplex) * bufsz, cudaMemcpyHostToDevice);
        cb->pop_front();

        // Run the FFT plan in-place (clobber input data).
        cufftExecC2C(plan, (cufftComplex *)(d_i), (cufftComplex *)(d_i), CUFFT_FORWARD);

        // Compute the mag squared in-place.
        calcPow<<<1, 512>>>(d_i, bufsz);

        // Sum to the accumulator
        calcSum<<<1, 512>>>((float *)d_i, d_a, bufsz);

        // Compare with max.
        calcMax<<<1, 512>>>((float *)d_i, d_x, bufsz);

        // If we've accumulated enough frames, copy back.
        if (cnt == accum)
        {
            // Average
            normalize<<<1, 512>>>(d_a, bufsz, (float)accum);

            cudaMemcpy(h_a, d_a, sizeof(float) * bufsz, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_x, d_x, sizeof(float) * bufsz, cudaMemcpyDeviceToHost);

            cudaMemset(d_a, 0, sizeof(float) * bufsz);
            cudaMemset(d_x, 0, sizeof(float) * bufsz);

            cnt = 0;
            fwrite(h_a, sizeof(float), bufsz, ofile);
            fwrite(h_x, sizeof(float), bufsz, ofile);

            std::cout << "." << std::flush;
        }
        else
        {
            cnt += 1;
        }
    }

    free(h_a);
    free(h_x);
    fclose(ofile);
};


void recv_to_file(
    uhd::usrp::multi_usrp::sptr usrp,
    const std::string &cpu_format,
    const std::string &wire_format,
    const std::string &channel,
    size_t samps_per_buff
)
{
    mdbuff_t buffs[num_buffers];
    //std::vector<void *> bufflist;

    for (uint32_t i = 0; i < num_buffers; i++)
    {
        buffs[i].data.resize(samps_per_buff);
        //bufflist.push_back(&buffs[i].front());
    }

    boost::circular_buffer<mdbuff_t *> cb(num_buffers);
    boost::thread gpu_thread(cuda_handler, &cb, samps_per_buff, 1000);
    uint32_t bufidx = 0;
    uint32_t overrun_cnt = 0;

    //create a receive streamer
    uhd::stream_args_t stream_args(cpu_format,wire_format);
    std::vector<size_t> channel_nums;
    channel_nums.push_back(boost::lexical_cast<size_t>(channel));
    stream_args.channels = channel_nums;
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);

    //std::vector<samp_type> buff(samps_per_buff);
    bool overflow_message = true;

    //setup streaming
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    stream_cmd.num_samps = size_t(0);
    stream_cmd.stream_now = true;
    stream_cmd.time_spec = uhd::time_spec_t();
    rx_stream->issue_stream_cmd(stream_cmd);

    // Run this loop until either time expired (if a duration was given), until
    // the requested number of samples were collected (if such a number was
    // given), or until Ctrl-C was pressed.
    while (not stop_signal_called) 
    {

        size_t num_rx_samps = rx_stream->recv(&buffs[bufidx].data.front(), buffs[bufidx].data.size(), buffs[bufidx].md, 3.0, false);

        if (buffs[bufidx].md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT)
        {
            std::cout << boost::format("Timeout while streaming") << std::endl;
            break;
        }

        if (buffs[bufidx].md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW)
        {
            if (overflow_message) 
            {
                overflow_message = false;
                std::cerr << boost::format(
                    "Got an overflow indication. Please consider the following:\n"
                    "  Your write medium must sustain a rate of %fMB/s.\n"
                    "  Dropped samples will not be written to the file.\n"
                    "  Please modify this example for your purposes.\n"
                    "  This message will not appear again.\n"
                ) % (usrp->get_rx_rate()*sizeof(sample_t)/1e6);
            }
            continue;
        }

        if (buffs[bufidx].md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE)
        {
            std::string error = str(boost::format("Receiver error: %s") % buffs[bufidx].md.strerror());
            std::cerr << error << std::endl;
            continue;
        }

        if (cb.full())
        {
            std::cerr << ++overrun_cnt << " Overrun(s)." << std::endl;
        }

        cb.push_back(&buffs[bufidx]);
        bufidx = (bufidx + 1) % num_buffers;

    }

    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    rx_stream->issue_stream_cmd(stream_cmd);

    std::cout << "Waiting for GPU thread... ";
    gpu_thread.join();
    std::cout << "Done." << std::endl;

}

typedef std::function<uhd::sensor_value_t(const std::string&)> get_sensor_fn_t;

bool check_locked_sensor(
    std::vector<std::string> sensor_names,
    const char* sensor_name,
    get_sensor_fn_t get_sensor_fn,
    double setup_time
) {
    if (std::find(sensor_names.begin(), sensor_names.end(), sensor_name) == sensor_names.end())
        return false;

    auto setup_timeout =
        std::chrono::steady_clock::now()
        + std::chrono::milliseconds(int64_t(setup_time * 1000));
    bool lock_detected = false;

    std::cout << boost::format("Waiting for \"%s\": ") % sensor_name;
    std::cout.flush();

    while (true) {
        if (lock_detected and
            (std::chrono::steady_clock::now() > setup_timeout)) {
            std::cout << " locked." << std::endl;
            break;
        }
        if (get_sensor_fn(sensor_name).to_bool()) {
            std::cout << "+";
            std::cout.flush();
            lock_detected = true;
        }
        else {
            if (std::chrono::steady_clock::now() > setup_timeout) {
                std::cout << std::endl;
                throw std::runtime_error(str(
                    boost::format("timed out waiting for consecutive locks on sensor \"%s\"")
                    % sensor_name
                ));
            }
            std::cout << "_";
            std::cout.flush();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << std::endl;
    return true;
}

int UHD_SAFE_MAIN(int argc, char *argv[])
{
    uhd::set_thread_priority_safe();

    //variables to be set by po
    std::string args, type, ant, subdev, ref, wirefmt, channel;
    size_t spb;
    double rate, freq, gain, setup_time;

    //setup the program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
        ("spb", po::value<size_t>(&spb)->default_value((1<<20)), "samples per buffer")
        ("rate", po::value<double>(&rate)->default_value(25e6), "rate of incoming samples")
        ("freq", po::value<double>(&freq)->default_value(200e6), "RF center frequency in Hz")
        ("gain", po::value<double>(&gain), "gain for the RF chain")
        ("ant", po::value<std::string>(&ant), "antenna selection")
        ("subdev", po::value<std::string>(&subdev), "subdevice specification")
        ("channel", po::value<std::string>(&channel)->default_value("0"), "which channel to use")
        ("ref", po::value<std::string>(&ref)->default_value("internal"), "reference source (internal, external, mimo)")
        ("wirefmt", po::value<std::string>(&wirefmt)->default_value("sc16"), "wire format (sc8, sc16 or s16)")
        ("setup", po::value<double>(&setup_time)->default_value(1.0), "seconds of setup time")
        ("skip-lo", "skip checking LO lock status")
        ("int-n", "tune USRP with integer-N tuning")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //print the help message
    if (vm.count("help")) {
        std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
        std::cout
            << std::endl
            << "This application streams data from a single channel of a USRP device to a file.\n"
            << std::endl;
        return ~0;
    }

    //create a usrp device
    std::cout << std::endl;
    std::cout << boost::format("Creating the usrp device with: %s...") % args << std::endl;
    uhd::usrp::multi_usrp::sptr usrp = uhd::usrp::multi_usrp::make(args);

    //Lock mboard clocks
    usrp->set_clock_source(ref);

    //always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("subdev")) usrp->set_rx_subdev_spec(subdev);

    std::cout << boost::format("Using Device: %s") % usrp->get_pp_string() << std::endl;

    //set the sample rate
    if (rate <= 0.0){
        std::cerr << "Please specify a valid sample rate" << std::endl;
        return ~0;
    }
    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate/1e6) << std::endl;
    usrp->set_rx_rate(rate);
    std::cout << boost::format("Actual RX Rate: %f Msps...") % (usrp->get_rx_rate()/1e6) << std::endl << std::endl;

    //set the center frequency
    if (vm.count("freq")) { //with default of 0.0 this will always be true
        std::cout << boost::format("Setting RX Freq: %f MHz...") % (freq/1e6) << std::endl;
        uhd::tune_request_t tune_request(freq);
        if(vm.count("int-n")) tune_request.args = uhd::device_addr_t("mode_n=integer");
        usrp->set_rx_freq(tune_request);
        std::cout << boost::format("Actual RX Freq: %f MHz...") % (usrp->get_rx_freq()/1e6) << std::endl << std::endl;
    }

    //set the rf gain
    if (vm.count("gain")) {
        std::cout << boost::format("Setting RX Gain: %f dB...") % gain << std::endl;
        usrp->set_rx_gain(gain);
        std::cout << boost::format("Actual RX Gain: %f dB...") % usrp->get_rx_gain() << std::endl << std::endl;
    }

    //set the antenna
    if (vm.count("ant")) usrp->set_rx_antenna(ant);

    std::this_thread::sleep_for(
        std::chrono::milliseconds(int64_t(1000 * setup_time))
    );

    //check Ref and LO Lock detect
    if (not vm.count("skip-lo")){
        check_locked_sensor(
            usrp->get_rx_sensor_names(0),
            "lo_locked",
            [usrp](const std::string& sensor_name){
                return usrp->get_rx_sensor(sensor_name);
            },
            setup_time
        );
        if (ref == "mimo") {
            check_locked_sensor(
                usrp->get_mboard_sensor_names(0),
                "mimo_locked",
                [usrp](const std::string& sensor_name){
                    return usrp->get_mboard_sensor(sensor_name);
                },
                setup_time
            );
        }
        if (ref == "external") {
            check_locked_sensor(
                usrp->get_mboard_sensor_names(0),
                "ref_locked",
                [usrp](const std::string& sensor_name){
                    return usrp->get_mboard_sensor(sensor_name);
                },
                setup_time
            );
        }
    }

    std::signal(SIGINT, &sig_int_handler);
    std::cout << "Press Ctrl + C to stop streaming..." << std::endl;

    //recv to file
    recv_to_file(usrp, "fc32", wirefmt, channel, spb);

    //finished
    std::cout << std::endl << "Done!" << std::endl << std::endl;

    return EXIT_SUCCESS;
}
