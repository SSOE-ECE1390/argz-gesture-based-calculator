# ARGZ Gesture-Based Calculator
ECE 2390 project: ARGZ Gesture-Based Calculator
## Project Description
The goal of this project is to develop a gesture-based calculator capable of performing basic mathematical operations such as addition, subtraction, multiplication, and division. The calculator will accept integer inputs (1 to 10) from a user by interpreting the user's hand gestures through live video and will perform the desired calculations as indicated by the user. This also includes hand position tracking so that gestures from only one user will be interpreted at a time.

## Team Members
* Zachary Waddell (zmw24@pitt.edu)
* Gabrielle Stokes (gos50@pitt.edu)
* Ram Goenka (rag334@pitt.edu)
* Allen Wang (axw12@pitt.edu)
## Milestones
* 09/22/2025 - Project planning finished
* 10/20/2025 - Basic MediaPipe gesture detection and finger counting working
* 11/03/2025 - Calculator functionality working
* 11/10/2025 - ROI cropping working
* 11/17/2025 - Project demonstration
* 12/01/2025 - Project due date

## Repository Structure
```
├── CODE_OF_CONDUCT.md
├── gesture_classifier_model.pkl
├── gesture_recognizer.task
├── LICENSE.md
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
   ├── data
   │   ├── Add
   │   ├── Divide
   │   ├── Minus
   │   └── Multiply
   ├── Dev_modules
   │   ├── __init__.py
   │   ├── AR_gab_gesture.py
   │   ├── detect_math_gesture.py
   │   ├── finger.py
   │   ├── gab_gesture.py
   │   ├── gab_main.py
   │   ├── gesture_recognition.py
   │   ├── gesture_training.py
   │   ├── roi_changes.py
   │   ├── training_pictures.py
   │   └── zmw_gesture_calculator.py
   ├── gesture_classifier_model.pkl
   ├── gesture_recognizer.task
   ├── main.py
   └── number.stl
```
## File Descriptions
This project contains a number of additional files that are used by GitHub to provide information and do tests on code.
### Markup files (*.md)
* README.md: Contains the information about the project such as the project objective, authors, repository structure, description of key files, and usage instruction.

* CODE_OF_CONDUCT.md: This file establishes a set of behavioral expectations for contributors and community members, promoting a positive and inclusive environment.

* LICENSE.md: This file specifies the licensing terms under which your project is released, informing users about how they can use, modify, and distribute your code.

## Running the project
1. Clone the project
```
git clone <repo-url> gesture_based_calculator
cd gesture_based_calculator
```
2. Create a Virtual Environment (Python 3.11)
```
python3.11 -m venv .venv
```
And activate it (for MacOS/Linus):
```
source .venv/bin/activate
```
For Windows
```
.venv\Scripts\activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Run the project