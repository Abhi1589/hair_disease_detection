import torch
from torchvision import transforms
from PIL import Image
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch.nn as nn
from torchvision import models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the classes for scalp conditions
classes = [
    'Alopecia Areata',
    'Contact Dermatitis',
    'Folliculitis',
    'Head Lice',
    'Lichen Planus',
    'Male Pattern Baldness',
    'Psoriasis',
    'Seborrheic Dermatitis',
    'Telogen Effluvium',
    'Tinea Capitis'
]

# Model accuracies (dummy values)
MODEL_ACCURACIES = {
    'vgg': 0.995,
    'custom_cnn': 0.9925,
    'mobilenet': 0.9892
}

# Additional information dictionaries
HOME_REMEDIES = {
    'Alopecia Areata': 'Try gentle scalp massage with essential oils.',
    'Contact Dermatitis': 'Avoid contact with irritants and use hypoallergenic products.',
    'Folliculitis': 'Maintain good hygiene and use anti-bacterial soaps.',
    'Head Lice': 'Use over-the-counter treatments and thoroughly clean combs and brushes.',
    'Lichen Planus': 'Use corticosteroid creams and avoid stress.',
    'Male Pattern Baldness': 'Consider using minoxidil or finasteride treatments.',
    'Psoriasis': 'Use moisturizing creams and avoid triggers like stress and smoking.',
    'Seborrheic Dermatitis': 'Use anti-fungal shampoos and avoid harsh hair products.',
    'Telogen Effluvium': 'Address underlying causes and maintain a balanced diet.',
    'Tinea Capitis': 'Use anti-fungal shampoos and consult a dermatologist.'
}

DOCTOR_CONTACTS = {
    'Alopecia Areata': 'Dr. John Doe - +1-555-1234',
    'Contact Dermatitis': 'Dr. Jane Smith - +1-555-5678',
    'Folliculitis': 'Dr. Emily Davis - +1-555-8765',
    'Head Lice': 'Dr. Michael Johnson - +1-555-4321',
    'Lichen Planus': 'Dr. Sarah Wilson - +1-555-6789',
    'Male Pattern Baldness': 'Dr. Robert Brown - +1-555-9876',
    'Psoriasis': 'Dr. Laura Taylor - +1-555-3456',
    'Seborrheic Dermatitis': 'Dr. Kevin Lee - +1-555-6543',
    'Telogen Effluvium': 'Dr. Patricia Martin - +1-555-7890',
    'Tinea Capitis': 'Dr. William Anderson - +1-555-2345'
}

DOCTOR_EMAILS = {
    'Alopecia Areata': 'alopecia@example.com',
    'Contact Dermatitis': 'contactdermatitis@example.com',
    'Folliculitis': 'folliculitis@example.com',
    'Head Lice': 'headlice@example.com',
    'Lichen Planus': 'lichenplanus@example.com',
    'Male Pattern Baldness': 'malepatternbaldness@example.com',
    'Psoriasis': 'psoriasis@example.com',
    'Seborrheic Dermatitis': 'seborrheicdermatitis@example.com',
    'Telogen Effluvium': 'telogeneffluvium@example.com',
    'Tinea Capitis': 'tineacapitis@example.com'
}

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model definitions
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, len(classes))  # Updated to match number of classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ModifiedVGG(nn.Module):
    def __init__(self, num_classes=len(classes)):  # Updated to use len(classes)
        super(ModifiedVGG, self).__init__()
        vgg = models.vgg16(weights=None)
        self.features = vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_models():
    # VGG
    try:
        vgg_model = ModifiedVGG()
        vgg_model.load_state_dict(torch.load('models/vgg.pth', map_location=device))
        vgg_model.to(device)
        vgg_model.eval()
        print("VGG model loaded successfully")
    except Exception as e:
        print(f"Error loading VGG model: {e}")

    # Custom CNN
    try:
        custom_model = CustomCNN()
        custom_model.load_state_dict(torch.load('models/custom_cnn.pth', map_location=device))
        custom_model.to(device)
        custom_model.eval()
        print("Custom CNN model loaded successfully")
    except Exception as e:
        print(f"Error loading Custom CNN model: {e}")

    # MobileNet
    try:
        mobilenet_model = models.mobilenet_v2()
        mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, len(classes))  # Updated to match number of classes
        mobilenet_model.load_state_dict(torch.load('models/mobilenet.pth', map_location=device))
        mobilenet_model.to(device)
        mobilenet_model.eval()
        print("MobileNet model loaded successfully")
    except Exception as e:
        print(f"Error loading MobileNet model: {e}")
    
    return {
        'vgg': vgg_model,
        'custom_cnn': custom_model,
        'mobilenet': mobilenet_model
    }

# Load all models
try:
    models_dict = load_models()
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    models_dict = None

@app.route('/')
def home():
    return render_template('index.html', accuracies=MODEL_ACCURACIES)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    model_type = request.form.get('model_type')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
        
    if file and model_type:
        try:
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess image
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Ensure the model type is valid
            if models_dict is None:
                return jsonify({'error': 'Models not loaded successfully'})
            if model_type not in models_dict:
                return jsonify({'error': 'Invalid model type selected'})

            # Get prediction
            model = models_dict[model_type]
            model = model.to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = classes[predicted.item()]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
                
            # Get additional information for the predicted condition
            home_remedy = HOME_REMEDIES.get(prediction, 'No specific remedy available')
            doctor_contact = DOCTOR_CONTACTS.get(prediction, 'No doctor contact available')
            doctor_email = DOCTOR_EMAILS.get(prediction, 'No email available')
                
            return jsonify({
                'success': True,
                'prediction': prediction,
                'confidence': f"{confidence:.2%}",
                'image_path': filepath,
                'accuracy': MODEL_ACCURACIES[model_type],
                'home_remedy': home_remedy,
                'doctor_contact': doctor_contact,
                'doctor_email': doctor_email
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
            
    return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)