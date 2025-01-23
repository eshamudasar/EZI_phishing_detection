# EZI Phishing Detection System

EZI Phishing Detection System is designed to identify phishing content in both Urdu and English languages. Leveraging machine learning algorithms, it provides users with tools to detect potential phishing threats effectively.

## Features

- **Keyword Phishing Detection:** Analyze text inputs to identify phishing-related keywords.
- **URL Phishing Detection:** Evaluate URLs to determine their legitimacy.
- **Real-Time Phishing Detection:** Offer instantaneous analysis of URLs for immediate threat assessment.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/eshamudasar/EZI_phishing_detection.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd EZI_phishing_detection
   ```

3. **Create a Virtual Environment:**
   ```bash
   python -m venv myenv
   ```

4. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source myenv/bin/activate
     ```

5. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Apply Migrations:**
   ```bash
   python manage.py migrate
   ```

2. **Run the Development Server:**
   ```bash
   python manage.py runserver
   ```

3. **Access the Application:**
   Open your web browser and navigate to `https://eziphish.com` to use the EZI Phishing Detection System.

## Project Structure

The repository is organized as follows:

- **`detection/`**: Contains the core application logic for phishing detection.
- **`ezi_phishing_detection_system/`**: Houses project-level settings and configurations.
- **`media/`**: Stores media files used within the application.
- **`staticfiles/`**: Includes static assets like CSS, JavaScript, and images.
- **`templates/`**: Contains HTML templates for rendering web pages.
- **`requirements.txt`**: Lists all the Python dependencies required for the project.

## Collaborators

This project is a collaborative effort among the following contributors:

- **Esha Mudasar**: [GitHub Profile](https://github.com/eshamudasar)
- **Ibrahim**: [GitHub Profile](https://github.com/Ibrhim12)
- **Zeba Khan**: [GitHub Profile](https://github.com/Zebakhan20)


## Acknowledgments

We extend our gratitude to all contributors and the open-source community for their invaluable support and resources.

