# Import the Hugging Face pipeline for emotion classification
from transformers import pipeline

# Load the pre-trained emotion detection model
# (This downloads the model the first time you run it)
emotion_classifier = pipeline(
    "text-classification", 
    model="cardiffnlp/twitter-roberta-base-emotion",
    return_all_scores=True  # Show all emotions with scores
)

# Get user input
print("\nWelcome to Emotion Detector! Type 'quit' to exit.\n")
while True:
    user_text = input("How are you feeling today? : ")
    
    if user_text.lower() == "quit":
        print("Goodbye! Take care :)")
        break
    
    # Analyze the emotion
    emotions = emotion_classifier(user_text)[0]  # Returns a list of emotions with scores
    
    # Print the top emotion
    top_emotion = max(emotions, key=lambda x: x['score'])  # Get emotion with highest score
    print(f"\nDetected Emotion: {top_emotion['label']} (Confidence: {top_emotion['score']:.2f})")
    
    # Optional: Print all emotions (uncomment to see)
    # print("\nDetailed Breakdown:")
    # for emotion in emotions:
    #     print(f"- {emotion['label']}: {emotion['score']:.2f}")