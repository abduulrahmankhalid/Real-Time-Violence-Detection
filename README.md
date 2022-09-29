# Real Time Violence Detection using MobileNet and Bi-directional LSTM
#### This repository is part of the Graduation Project for NTI Training -  AI and IoT DEY Initiative. 
### Kaggle Notebook : https://www.kaggle.com/code/abduulrahmankhalid/real-time-violence-detection-mobilenet-bi-lstm

## Understanding the Dataset
This dataset contain ONLY two directories: *NonViolence* (which contains 1000 real life situations videos like eating, sports activity, singing, etc and this directory doesn't have any violence situations) and the other directory *Violence* (contains 1000 videos with severe violence in various situations).

## Preprocessing

All what we need to do here is to extract the frames from all the vedios.

We found this helpful function on internet (it is easy to understand) that extract the frames from vedios for us.


def frames_extraction(video_path):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''
 
    # Declare a list to store video frames.
    frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)
 
    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
 
    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):
 
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
 
        # Reading the frame from the video. 
        success, frame = video_reader.read() 
 
        # Check if Video frame is not successfully read then break the loop
        if not success:
            break
 
        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)
    
    # Release the VideoCapture object. 
    video_reader.release()
 
    # Return the frames list.
    return frames_list


## Model Building

#### Metrics considered for Model Evaluation
*Accuracy, Precision, Recall and F1 Score*
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

### Architectures
Here we built many models with different Architectures as we were searching for the best model.
- *Convnet + Bidirectional-LSTM* accuracy: 90% - 92%

- *Agg-16 + Bidirectional-LSTM* accuracy: 84% - 86%
- *MobileNet V2 + Bidirectional-LSTM* accuracy: 94% - 97%
