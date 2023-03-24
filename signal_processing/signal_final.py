#this function is just for testing, imports the libraries for use, probably can ignore as far as UI merging
import utils
import mne 
import numpy
import matplotlib.pyplot as plt
import numpy # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import muselsl_utils as muselsl_utils  # Our own utility functions
import utils
import scipy.stats
import statistics
import time
import pandas
import pylsl
print("imported")

def dominant_freq(window):

    #performs a power spectral display function on both the alpha and theta range
    psd_raw_alpha=numpy.array(mne.time_frequency.psd_multitaper(window,fmin=8,fmax=13,picks=["AF7","AF8","TP9","TP10"]))
    psd_raw_theta=numpy.array(mne.time_frequency.psd_multitaper(window,fmin=4,fmax=8,picks=["AF7","AF8","TP9","TP10"]))

#calculates dominant frequency by collapsing power spectral density across all 4 channels and taking the frequency with the largest sum
####################################################################
    store = numpy.sum(psd_raw_alpha[0], axis=0)
    max_index=numpy.argmax(store)
    dominant_alpha= psd_raw_alpha[1]
    dominant_alpha= dominant_alpha[max_index]

#performs the same as above on the theta range
####################################################################
    store = numpy.sum(psd_raw_theta[0], axis=0)
    max_index=numpy.argmax(store)
    dominant_theta= psd_raw_theta[1]
    dominant_theta= dominant_theta[max_index]
    return [dominant_alpha, dominant_theta]


#user will be the patient that we are working with, this will generate a distribution of dominant frequencies
#then we can work with the descriptive statistical measures from this distribution
def baseline(user):

    #defines some parameters to be used by the MNE library, specific to our recording setup
    info = mne.create_info(["timestamps","TP9","AF7","AF8", "TP10", "Right AUX"], 256, ch_types="eeg")

    #imports the raw recording data from the Data folder, 
    raw_recording = pandas.read_csv(str("Data/"+user+"_Baseline.csv"))
    print("Data imported...")
    #takes the raw recording and moves it into an MNE raw array object, allows for easier manipulation later
    #this is a different transformation than later as we are working with a Pandas object
    raw_array = mne.io.RawArray(raw_recording.values.T/1000000, info)
    print("Raw moved to raw_array...")
    #note that we are working with a setup recording at 256 Hz, thus a 256 time step sample represents one second
    step=0 #sets the start point of our window
    max=len(raw_array) #sets the max sample point of our recording
    distribution=[]
    alpha_dist = []
    theta_dist = []
    while step<max:
        #takes an eight second sample from the raw recording
        raw_window=raw_array.get_data(picks=None, start=step,stop=(step+2048))
        #again moves this array into ane MNE raw array object            
        mne_window = mne.io.RawArray(raw_window, info)
        #advances the sample window forward by a second
        step=step+2048
        try:
            doms = dominant_freq(mne_window)
            alpha_dist.append(round(doms[0],3))
            theta_dist.append(round(doms[1],3))
        
        except ValueError:
            print("distributions generated...")


    return (alpha_dist, theta_dist)
        
        
alpha_dist, theta_dist = baseline("JacobLong1")


#this function calculates the dominant frequency for some gien window, takes an MNE raw array as input
#returns a the dominant frequency in the alpha range as well as the dominant frequency in the theta range

def live_run():
    class Band:
        Delta = 0
        Theta = 1
        Alpha = 2
        Beta = 3

    blink_types = ["Right Wink", "Left Wink", "Normal Blink", "Nothing"]

    """ EXPERIMENTAL PARAMETERS """
    # Modify these to change aspects of the signal processing

    # Length of the EEG data buffer (in seconds)
    # This buffer will hold last n seconds of data and be used for calculations
    BUFFER_LENGTH = 10

    # This is the batch size that out code will grab from the buffer when it is ready to process the new data
    EPOCH_LENGTH = 4

    # Amount of overlap between two consecutive epochs (in seconds)
    OVERLAP_LENGTH = 0

    # Amount to 'shift' the start of each next consecutive epoch
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

    # Index of the channel(s) (electrodes) to be used
    # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
    INDEX_CHANNEL = [0, 1, 2, 3]

    if __name__ == "__main__":

        """ 1. CONNECT TO EEG STREAM """

        # Search for active LSL streams
        print('Looking for an EEG stream...')
        streams = pylsl.resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('Can\'t find EEG stream.')

        # Set active EEG stream to inlet and apply time correction
        print("Start acquiring data")
        inlet = StreamInlet(streams[0], max_chunklen=12)
        eeg_time_correction = inlet.time_correction()

        # Get the stream info and description
        info = inlet.info()
        description = info.desc()

        # Get the sampling frequency
        # This is an important value that represents how many EEG data points are
        # collected in a second. This influences our frequency band calculation.
        # for the Muse 2016, this should always be 256
        fs = int(info.nominal_srate())

        """ 2. INITIALIZE BUFFERS """

        # Initialize raw EEG data buffer - 256*5 x 4 array to store the last 5 seconds
        eeg_buffer = numpy.zeros((int(fs * BUFFER_LENGTH), 4))
        filter_state = None  # for use with the notch filter

        # Compute the number of epochs in "buffer_length"
        n_win_test = int(numpy.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                SHIFT_LENGTH + 1))

        # Initialize the band power buffer (for plotting)
        # bands will be ordered: [delta, theta, alpha, beta]
        band_buffer = numpy.zeros((n_win_test, 4))

        """ 3. GET DATA """
        info = mne.create_info(["TP9","AF7","AF8", "TP10"], 256, ch_types="eeg")
        # The try/except structure allows to quit the while loop by aborting the
        # script with <Ctrl-C>
        print('Press Ctrl-C in the console to break the while loop.')

        a_store=[]
        t_store=[]
        try:
            counter=0
            # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
            while True:
                time.sleep(4)
                """ 3.1 ACQUIRE DATA """
                # Obtain EEG data from the LSL stream
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))


                # Only keep the channel we're interested in
                ch_data = numpy.array(eeg_data)[:, 0:4]

                # Update EEG buffer with the new data
                #this eeg_buffer is a 5 second store of data, sampled at 256 Hz, it is a numpy array
                eeg_buffer, filter_state = muselsl_utils.update_buffer(
                    eeg_buffer, ch_data, notch=False,
                    filter_state=filter_state)
            
                if eeg_buffer[0,0] != 0:

                    """ 3.2 COMPUTE BAND POWERS """
                    # Get newest samples from the buffer
                    data_epoch = muselsl_utils.get_last_data(eeg_buffer,
                                                EPOCH_LENGTH * fs)

                    #had to transpose to process with MNE package
                    data_epoch = numpy.transpose(data_epoch)

                    #transforms the pulled epoch from an np array to an MNE epoch object
                    mne_epoch = mne.io.RawArray(data_epoch, info)

                    #calls upon previously defined function to pull dominant frequencies for the chunk
                    freqs = dominant_freq(mne_epoch)
                    
                    #if the store is 60 samples long (arbitrary number?)
                    if len(a_store) == 60:
                        #remove first sample
                        a_store.pop(0) 
                        #add most recent generated sample to end of store
                        a_store.append(round(freqs[0],3))
                    else:
                        #add most recent generated sample to end of store
                        a_store.append(round(freqs[0],3))
            
                    #same for theta range
                    if len(t_store) == 60:
                        t_store.pop(0) 
                        t_store.append(round(freqs[0],3))
                    else:
                        t_store.append(round(freqs[1],3))
                
                #if we have sufficient samples, perform a kruskal wallace test on each set of dominant frequencies
                if len(a_store)==60: ####FIX THIS IN POST
                    ap = scipy.stats.mannwhitneyu(x=a_store,y=alpha_dist, use_continuity=True, alternative='less', axis=0, method='auto')
                    tp = scipy.stats.mannwhitneyu(x=t_store,y=theta_dist, use_continuity=True, alternative='less', axis=0, method='auto')
                    if ap[1]+tp[1] < 0.05:
                        print("The patient is experiencing nicotwithdrawal, p-val = ", ap[1]+tp[1])
                        state = 1
                    else:
                        print("The patient is not experiencing nicotwithdrawal, p-val = ", ap[1]+tp[1])
                        state = 0 
        except KeyboardInterrupt:
            #serial_port.close()
            print('Closing!')
            return None
        
alpha_dist, theta_dist = baseline("JacobLong1")
live_run()

