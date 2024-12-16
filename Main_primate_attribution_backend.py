import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import time
import Main_reasoning_backend
from Main_reasoning_backend import explain_model_decisions
import Main_attribution_snomed_backend
from Main_attribution_snomed_backend import search_snomed 

import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
import torch
output_file = "explanations.json"
output_file = "diagnosis_attribution.json"
snomed_csv_file = "G:/USA Projects/attribution_classic/work/updating_snomed/data/snomed_ctid.csv" # Path to your SNOMED CT CSV file

# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    mse_loss = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts.float())
            _, predicted = torch.max(outputs.data, 1)

            # Update the number of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store labels and predictions for metrics calculation
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

            # Calculate MSE loss
            mse_loss += mean_squared_error(labels.numpy(), predicted.numpy())

    # Calculate accuracy
    accuracy = correct / total

    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate Mean Squared Error (MSE)
    mse_loss = mse_loss / len(test_loader)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Mean Squared Error (MSE): {mse_loss:.4f}')

    return accuracy, precision, recall, f1, mse_loss





# Load JSON file
def load_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    print("Primate Dataset_Training Data Loaded Successfully")
    return data

# Load CSV file and ensure it reads the first row as header
def load_csv(csv_file):
    print("PHQ9 Ontology Data Loaded Successfully")
    return pd.read_csv(csv_file, header=0)

# Create a dataset for the neural network
class PHQ9Dataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Function to create embeddings using Sentence Transformers
def create_embeddings(texts, tokenizer, model, batch_size=32):
    """
    Create embeddings for a list of texts in batches to optimize memory usage.
    
    Args:
        texts (list): List of text strings.
        tokenizer: Tokenizer for the Sentence Transformer model.
        model: Sentence Transformer model for embeddings.
        batch_size (int): Batch size for processing.
    
    Returns:
        torch.Tensor: Embeddings for the input texts.
    """
    embeddings = []
    model.eval()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            batch_embeddings = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
        embeddings.append(batch_embeddings)
    
    return torch.cat(embeddings)

# Main function to orchestrate the workflow
def main(json_file, csv_file):
    # Load data
    json_data = load_json(json_file)
    phq9_df = load_csv(csv_file)

    # Prepare data for training
    questions = phq9_df.columns[1:]  # Skip the first column (assumed to be an index or irrelevant)
    all_texts = []
    all_labels = []
    index = 0
    alpha =0

    # Iterate through each post_text and classify based on the PHQ-9 questions
    for entry in json_data:
        print("Primate _Dataset_Iterations Count--->:", index)
        post_text = entry['post_text']
        for question in questions:
            print("Primate _Dataset_training_questions Count--->:", alpha)
            alpha = alpha+1
            symptoms = phq9_df[question].dropna().tolist()  # Extract relevant symptoms
            for symptom in symptoms:
                if symptom in post_text:
                    all_texts.append(post_text)
                    all_labels.append(question)
        index += 1
    print("System Under training")
    print("Hold on ")

    # Encode labels
    le = LabelEncoder()
    all_labels_encoded = le.fit_transform(all_labels)

    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(all_texts, all_labels_encoded, test_size=0.2)
    X_train = X_train[:80]  # Use only the first 20 samples
    y_train = y_train[:80]
    X_test = X_test[:80]
    y_test = y_test[:80]

    # Load Sentence Transformers model and tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence_model = AutoModel.from_pretrained(model_name)

    # Create embeddings for the training and test data
    train_embeddings = create_embeddings(X_train, tokenizer, sentence_model, batch_size=4)
    test_embeddings = create_embeddings(X_test, tokenizer, sentence_model, batch_size=4)

    # Create Dataset objects
    train_dataset = PHQ9Dataset(train_embeddings, y_train)
    test_dataset = PHQ9Dataset(test_embeddings, y_test)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize neural network
    input_dim = train_embeddings.shape[1]  # Size of the embedding vector
    output_dim = len(le.classes_)  # Number of unique PHQ-9 questions
    model = SimpleNN(input_dim, output_dim)

    # Set up training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the neural network
    model.train()
    for epoch in range(10):  # Number of epochs
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        
    print("System Training Complete")
    # Evaluate the model
    
    print("System Testing begins..................")
    print("Hold your breath ")
     # Define the criterion (loss function) for training
    criterion1 = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    
    criterion2 = nn.BCEWithLogitsLoss()
    
    #you can set criterion 
    
    # Example of calling the function
    system_accuracy, precision, recall, f1, mse_loss = evaluate_model(model, test_loader, criterion)
    print("system_accuracy",system_accuracy )
    
    
    

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print("System Tested with test data successfully")
    #print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    # Save the model with a unique name
    unique_model_name = f"PHQ9_Classifier_{int(time.time())}.pth"
    torch.save(model.state_dict(), unique_model_name)
    print(f"Model saved as {unique_model_name}")
    
    
    input_dim = train_embeddings.shape[1]  # Same as during training
    output_dim = len(le.classes_)  # Same as during training
    
    
    
        # Save the model with a unique name
    unique_model_name = f"PHQ9_Classifier_{int(time.time())}.pth"
    torch.save(model.state_dict(), unique_model_name)
    #print(f"Model saved as {unique_model_name}")

    # Load the saved model
    loaded_model = SimpleNN(input_dim, output_dim)
    loaded_model.load_state_dict(torch.load(unique_model_name))
    loaded_model.eval()  # Set to evaluation mode

    # Example: Generating the PHQ-9 checklist
    user_input = "I posted yesterday, but things keep getting worse.\n\nSo my spouse and I were given a 30 day move out notice because our depression mess was too much for them. It\u2019s not even that\u2019s bad, and we\u2019ve been actively trying to get better, but they wouldn\u2019t work with us at all.\n\nMy spouse called her mom today to wish her a happy birthday and her mom ended up asking a bunch of questions about our plan and what we\u2019re doing, and my fianc\u00e9e broached the subject of her possibly taking our cats. Her mom proceeded to freak out at her and say that she was being disrespectful and then yelled at my spouse for being manipulative when she started to cry. She then said that if we really can\u2019t find anything that we can move in with her, but obviously that\u2019s not something that any of us want. \n\nMy spouse is currently sobbing in the bathroom, barely able to move, and I just feel so utterly fucking helpless and afraid. She told me that she wants to kill herself so fucking bad but that she can\u2019t because that would be unfair to me. I worry every fucking day that this will be the day that her will to live outweighs that guilt she feels for leaving me. I have for a long time. I used to come home when I worked an earlier shift from her and check the closet to make sure I didn\u2019t find her hanging there, and that was even when things were going relatively okay.\n\nI don\u2019t know how we\u2019re supposed to do this. How do we get an apartment if we\u2019re being kicked out in the middle of our lease? No one will want us with this on our record. I\u2019m terrified of if we have to go to her mom\u2019s. I would suggest that just she move back in (because I think part of her mom\u2019s issue is with me) and I find a couch to crash on until we can figure it out, but I worry that her mental state would get worse being at her mom\u2019s without me, dealing with her mother\u2019s emotional abuse every day.\n\nI\u2019m terrified. I feel like I\u2019m spiraling out of control. My spouse\u2019s mental health is so much worse than mine so I\u2019m the one who always has to be strong and supportive, but I\u2019m losing it. I\u2019m losing my grip on things. I don\u2019t have anyone to go to. I don\u2019t really have many friends because my depression worsened after college and I lost touch with almost everyone I was close to. I\u2019m going to crack. I\u2019m afraid. I\u2019m so fucking afraid."
    result = generate_phq9_checklist(user_input, tokenizer, sentence_model, loaded_model, le)
    print(result)
    
    
    print("------------------------")
    
    
     # Print the results
    print("\nAI-Generated PHQ-9 Checklist:")
    for question, response in result["checklist"].items():
        print(f"{question}: {response}")

    print(f"\nDiagnosis: {result['diagnosis']}")

    
    
    
    explanations = explain_model_decisions(user_input, model, tokenizer, result["checklist"], result['diagnosis'], test_loader, criterion)
    # Save explanations to a JSON file
    
    with open(output_file, "w") as file:
        json.dump(explanations, file, indent=4)
    
    print(f"Explanations saved to {output_file}")
    print(json.dumps(explanations, indent=4))
    
    ai_generated_diagnosis = result['diagnosis']
    # Search for the term in SNOMED CT
    snomed_result = search_snomed(ai_generated_diagnosis, snomed_csv_file)
    
    # Save the result to a JSON file
    with open(output_file, "w") as json_file:
        json.dump(snomed_result, json_file, indent=4)
    
    print(f"Attribution results saved to {output_file}.")

    
    
    
    
    
    
    
    return unique_model_name


def generate_phq9_checklist(user_input, tokenizer, sentence_model, model,label_encoder ):
    """
    Generate an AI-based PHQ-9 checklist and diagnosis.
    
    Args:
        user_input (str): User-provided text input.
        tokenizer: Tokenizer for the Sentence Transformer model.
        sentence_model: Sentence Transformer model for embeddings.
        model: Trained classifier.
        label_encoder: Label encoder for decoding predictions.
    
    Returns:
        dict: PHQ-9 checklist with identified symptoms and diagnosis.
    """
    # Generate embeddings for the user input
    
    
    input_embedding = create_embeddings([user_input], tokenizer, sentence_model)
    
    # Pass the embeddings through the model
    model.eval()
    with torch.no_grad():
        outputs = model(input_embedding.float())
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    # Decode the predicted class to the PHQ-9 question
    predicted_question = label_encoder.inverse_transform([predicted_class])[0]
    
    # Create the checklist
    checklist = {question: "No" for question in label_encoder.classes_}
    checklist[predicted_question] = "Yes"
    
    # Generate diagnosis
    positive_symptoms = sum(1 for response in checklist.values() if response == "Yes")
    diagnosis = (
        "Minimal depression" if positive_symptoms <= 4 else
        "Mild depression" if positive_symptoms <= 9 else
        "Moderate depression" if positive_symptoms <= 14 else
        "Moderately severe depression" if positive_symptoms <= 19 else
        "Severe depression"
    )
    
    return {
        "checklist": checklist,
        "diagnosis": diagnosis
    }




if __name__ == "__main__":
    print("...Welcome...")
    print("System Training begins---------->")
    unique_model_name = main("G:/USA Projects/attribution_classic/work/data/primate_dataset.json", "G:/USA Projects/attribution_classic/work/data/PHQ9Questions_depression_ontology.csv")
    
    print("------------------System Diagnosis Complete----------------")
    print("-----------------------------------------------------------")
    print("Model  has been successfully trained to classify user input into AI generated PHQ9 Checklist")
    print("Model name:", unique_model_name)
    print("This version is updted on 6 December 2024")
    
    
    
    print("Reasoning complete ")
    print("Snomed Attibution Complete ")
    print("now working on front end")
    
    #user_input = "I feel sad and hopeless almost every day, tired, sleepless, and I don't enjoy activities I used to like."
# checklist = {
#     "Concentration trouble, addiction": "No",
#     "Feeling down, depressed, or hopeless, Advice related": "No",
#     "Feeling or emotion words (-ve)": "Yes",
#     "Little interest or pleasure": "No",
#     "Moving or speaking very slowly, restless": "No",
#     "Pain, hurt or death": "No",
#     "Poor appetite or overeating": "No",
#     "Small or too much sleep": "No",
#     "Tired or little energy or Time related": "No",
# }
# diagnosis = "Minimal depression"

# Get explanations



    
    #sentence_model = AutoModel.from_pretrained("bert-base-uncased")
    # Generate the AI-based PHQ-9 checklist and diagnosis
   

    #result = generate_phq9_checklist(user_input, AutoTokenizer, unique_model_name)

   
