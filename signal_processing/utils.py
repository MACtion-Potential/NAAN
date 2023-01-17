# A file containing some useful functions for eyeblink detection

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def find_eye_events(dataframe, number_of_blinks, duration_of_blink=1, sampling_frequency=256, duration_before_peak=0.25, jitter=0):
    """
    A custom function for finding eyeblinks in our data, where we know roughly the total number of blinks that should occur.
    
    Arguments:
        dataframe: A dataframe representing the raw data collected with the Muse
        number_of_blinks: The number of blinks in the data
        duration_of_blink: The duration in seconds of a blink event
        sampling_frequency: The sampling frequency of the Muse headband. For Muse 2016, this is about 256 Hz
        duration_before_peak: The interval between the start of a window and the eyeblink peak.
        jitter: Whether to randomize the value of `duration_before_peak` that gets used, between 0 and the value provided.
                If you're just trying to find eyeblinks, set this to `False`.
                If you're trying to test the robustness of your classifier against staggered windows, set it to True
    Returns:
        blink_locations: An array containing the start and end indices of eyeblinks.

    Rationale:
        Since the blinks are usually the tallest peaks in the data, we can pull out the `number_of_blinks` tallest peaks in the data.

        However, sometimes eyeblinks are not smooth, but contain a few sub-peaks. To avoid counting an eyeblink twice just because it gives a spiky signal, we will count only the tallest point of the eyeblink, and then refuse to count anything within a `duration_of_blink` range around that peak as being another eyeblink.
        Since our data has 4 channels, we can look through all 4 channels to find the next highest peak

        The scipy and mne functions available for this are good, but they are made for more general cases where you don't know the number of blinks beforehand. As a result, they require you to tune some parameters to tune your selectivity and sensitivity to eyeblinks. This is tedious to do for many datasets, so a custom function is the way to go.
    """
    # A list of all the channels we want to check
    channel_list = ["TP10", "TP9", "AF8", "AF7"]
    # Get the array from our dataframe
    data_array = abs(dataframe[channel_list].values - dataframe[channel_list].values.mean(axis=0))
    # Initialize our array to store our output
    blink_locations = []
    # Count through all the blinks we need to find
    for blink_index in range(number_of_blinks):
        # Find the number of points from the start of a blink to the peak, and from the peak to the end
        offset = jitter*(np.random.rand() - 0.5)*2
        duration_before_peak += offset
        start_to_peak = int(duration_before_peak*sampling_frequency)
        peak_to_end = int(duration_of_blink*sampling_frequency) - start_to_peak
        # Get the location of the next tallest peak, across all channels
        #print(data_array)
        #print(type(data_array[0,0]))
        blink_location = np.argmax(data_array)
        #print(blink_location)
        blink_location = np.unravel_index(blink_location, data_array.shape)[0]
        #print(blink_location)
        #print()
        # If we can form a full window around that blink, add it to our data
        if blink_location - start_to_peak >= 0 and blink_location + peak_to_end < data_array.shape[0]:
            blink_locations.append([blink_location - start_to_peak, blink_location + peak_to_end])
        # Set the data around that blink location to 0 so we don't count it again
        data_array[blink_location-start_to_peak:blink_location+peak_to_end,:] = 0
    #     plt.plot(data_array[:,0])
    # plt.show()
    # Return our result
    return np.array(blink_locations)

def find_noneye_events(dataframe, number_of_blinks, duration_of_blink=1, sampling_frequency=256, duration_before_peak=0.25, jitter=0):
    """
    A custom function to find stretches of data that are not eyeblinks. It works by using `find_eye_events()` to obtain the eye event locations, then going through the data and extracting windows that have no overlap with eyeblinks.

    For more information on how the eye events are detected and how the parameters are used, see the description for `find_eye_events()`
    """
    # Get the blinks in this dataset
    eye_events = find_eye_events(dataframe, number_of_blinks, duration_of_blink, sampling_frequency, duration_before_peak, jitter)
    # Get the width of a window
    window_width = int(duration_of_blink*sampling_frequency)
    # Initialize a container for the indices that mark the start and end of non-blink windows
    nonblink_locations = []
    # Initialize the indices definining the winow we are currently looking at
    window_start_index = 0
    window_end_index = 0
    # Idea: Check all possible windows, by shifting over by one each time. If we find a window that 
    # doesn't contain an eyeblink, shift up by a whole window width so that the next non-blink window won't overlap.
    while window_start_index < (dataframe.values.shape[0] - window_width):
        # Update the ending index of the window
        window_end_index = window_start_index + window_width
        # Set a flag claiming that this window contains no blink. If we're wrong,
        # we'll find out in this next loop and change the value of this variable to True.
        # if we're right though, then this variable won't change.
        overlaps_with_blink = False
        # GO through all the blinks/winks in the data
        for eye_event_index in range(number_of_blinks):
            # Get the starting and ending indices for the blink
            blink_start_index = eye_events[eye_event_index, 0]
            blink_end_index = eye_events[eye_event_index, 1]
            # Check if this blink/wink is in our current window
            if not (blink_end_index < window_start_index or blink_start_index > window_end_index):
                # If so, note it down and stop looking. This window is not a non-blink/wink window
                overlaps_with_blink = True
                break
        # If this window didn't overlap with any winks or blinks, then append its start and end indices to
        # our output, and shift up our window start index by a whole window width to avoid overlap
        if not overlaps_with_blink:
            nonblink_locations.append(
                [window_start_index, window_end_index]
            )
            window_start_index += window_width
        # If it did overlap with a blink, advance our window by one index
        else:
            window_start_index += 1
    return np.array(nonblink_locations)


def prepare_data(user_dataset, duration_of_blink=1, sampling_frequency=256, duration_before_peak=0.25, jitter=0):
    X = []
    Y = []
    for key in user_dataset:
        # Get the dataframe for this recording
        dataframe = user_dataset[key].to_data_frame()
        # Get an estimate for the number of eyeblinks, subtract one to be conserative
        recording_duration = float(key.split("_")[2].split("s")[0])
        blink_frequency = float(key.split("_")[3].split("s")[0])
        number_of_blinks = int(recording_duration/blink_frequency) - 2
        # Plot it so we can see how good the detection is
        dataset_type = key.split("_")[1]

        # For the datasets where we are interested in training on the eye events, go through all the eye events
        # and create the feature vector we need
        if dataset_type in ("LeftWinks", "RightWinks", "NormalBlinks"):
            # Get the eye events in the recording
            eye_events = find_eye_events(dataframe, number_of_blinks, duration_of_blink, sampling_frequency, duration_before_peak, jitter)
            #print(f"{dataset_type}: {eye_events.shape}")
            # if dataset_type == "NormalBlinks": assert False
            # For each eye event, plot, and get the feature vector and label
            for eye_event_idx in range(eye_events.shape[0]):
                X.append(dataframe[["AF7", "AF8", "TP9", "TP10"]].iloc[eye_events[eye_event_idx, 0]:eye_events[eye_event_idx, 1]])
                if dataset_type == "RightWinks": Y.append(0)
                elif dataset_type == "LeftWinks": Y.append(1)
                elif dataset_type == "NormalBlinks": Y.append(2)
        # Otherwise, if it's a dataset where we want to train on the non-blinks, go through all 
        elif dataset_type in ("FewBlinks", "NoBlinks"):
            # Get the eye events in the recording
            noneye_events = find_noneye_events(dataframe, number_of_blinks, duration_of_blink, sampling_frequency, duration_before_peak, jitter)
            # For each non eye event, plot, and get the feature vector and label
            for noneye_event_idx in range(noneye_events.shape[0]):
                X.append(dataframe[["AF7", "AF8", "TP9", "TP10"]].iloc[noneye_events[noneye_event_idx, 0]:noneye_events[noneye_event_idx, 1]])
                if dataset_type == "FewBlinks": Y.append(3)
    Y = np.array(Y)
    return X, Y

def compute_features(unfiltered_data_array, filter=None, pca=None, use_original=False):
    """
    Computes the features for the classifier.

    Arguments:
        data_array: A 256 data_array 4 array representing a sampling from all 4 channels across 256 time points.
        filter: A filter to apply to the data, via scipy.signal.lfilter. Must be a (b, a) tuple. (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter)
        pca: An instance of a principle component analysis object to apply to the data.
        use_original: Whether to compute features for the original, unfiltered, un-PCA'd data.

    Returns:
        A vector representing the feature vector for this window of data.
    """
    # Create the versions of the data
    # Initialize the feature vector
    data_array_versions = []
    if use_original: data_array_versions.append(unfiltered_data_array)

    feature_vector = []

    if filter != None:
        b, a = filter
        filtered_data_array = signal.lfilter(b, a, unfiltered_data_array, axis=1)
        data_array_versions.append(filtered_data_array)

    if pca != None:
        pca_data_array = pca.transform(np.array([unfiltered_data_array,]))
        data_array_versions.append(pca_data_array)
    
    for data_array in data_array_versions:

        for channel_1 in range(4):
            mean_1 = data_array[:,channel_1].mean()
            channel_1_adjusted = data_array[:,channel_1] - mean_1
            # feature_vector.append((channel_1_adjusted).max())
            # feature_vector.append((channel_1_adjusted).min())
            # feature_vector.append((channel_1_adjusted[1:] - channel_1_adjusted[:-1]).mean())
            # feature_vector.append(data_array[:,channel_1].std())
            # feature_vector.append(np.sqrt(np.mean(np.square(data_array[:,channel_1]))))
            for channel_2 in range(channel_1+1, 4):
                mean_2 = data_array[:,channel_2].mean()
                # mean_2 = 0
                # channel_2_adjusted = data_array[:,channel_2] - mean_2
                # feature_vector.append(channel_1_adjusted.max() - channel_2_adjusted.max())
                # feature_vector.append(data_array[:,channel_1].min() - data_array[:,channel_2].min())
                # difference = channel_1_adjusted - channel_2_adjusted
                # feature_vector.append(difference.max())
                # feature_vector.append(difference.min())
                # feature_vector.append(np.sqrt(np.mean(np.square(difference), axis=0)))
                # feature_vector.append(difference.mean())
                # feature_vector.append((channel_1_adjusted[1:] - channel_2_adjusted[:-1]).max())
                # feature_vector.append(whatever you want to add)
                # feature_vector.append(
                # np.cov(data_array[:,channel_1], data_array[:,channel_2])[0,1]
                feature_vector.append((data_array[:, channel_1] - data_array[:, channel_2]).max())
                # )
    return feature_vector

def create_butterworth_filter(cutoffs, fs, filter_type="lowpass", order=5):
    assert (len(cutoffs) == 1 and filter_type in ("lowpass", "highpass")) or (len(cutoffs) == 2 and filter_type in ("bandpass", "highpass"))
    nyq = 0.5 * fs
    if len(cutoffs) == 1:
        normal_cutoffs = cutoffs[0] / nyq
    elif len(cutoffs) == 2:
        normal_cutoffs = (cutoffs[0]/nyq, cutoffs[1]/nyq)
    b, a = signal.butter(order, normal_cutoffs, btype=filter_type, analog=False)
    return b, a