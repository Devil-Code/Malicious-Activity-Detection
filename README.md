# Malicious Detection

Malicious Detection is a project focused on identifying and classifying malicious network traffic using machine learning techniques. This project preprocesses the KDD dataset and applies a Random Forest classifier to detect different types of network attacks.

## Table of Contents

- [Description](#description)
- [Screenshot](#screenshot)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors and Acknowledgment](#authors-and-acknowledgment)
- [Technologies Used](#technologies-used)
- [Contact Information](#contact-information)

## Description

This project aims to detect malicious activities within network traffic data by preprocessing the KDD dataset and using a Random Forest classifier for classification. The dataset undergoes various preprocessing steps including data cleaning, encoding categorical variables, and feature engineering.

## Screenshot

![Alt Text](/screenshots/accuracy.png)

## Features

- **Data Cleaning**: Removing unnecessary columns and handling missing values.
- **Data Encoding**: Encoding categorical variables into numerical values.
- **Feature Engineering**: Transforming and generating relevant features for the model.
- **Model Training**: Training a Random Forest classifier to detect malicious activities.
- **Model Evaluation**: Evaluating the model performance using accuracy, confusion matrix, and classification report.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Devil-Code/Malicious-Activity-Detection.git
    ```

2. Navigate to the project directory:
    ```bash
    cd malicious-detection
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ``

4. Ensure you have the KDD dataset file (`kdd.csv`) in the project directory.

## Usage

To use this project, follow these steps:

1. Run the Jupyter Notebook:
    ```bash
    jupyter notebook Malicious_Detection.ipynb
    ```

2. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

3. Alternatively, run the Python script if available:
    ```bash
    python malicious_detection.py
    ```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Add some feature"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-name
    ```
5. Create a new Pull Request.

## License

This project is licensed under the GNU GPL v3.0 License - see the [LICENSE](LICENSE) file for details.

## Authors and Acknowledgment

- **Pritesh Gandhi** - *Initial work* - [YourGitHubProfile](https://github.com/Devil-Code)
- Acknowledgments to [Colab](https://colab.research.google.com/) for providing an excellent platform for running Jupyter Notebooks.

## Technologies Used

- **Python**: Programming language
- **NumPy**: Numerical computing library
- **Pandas**: Data manipulation and analysis library
- **Scikit-learn**: Machine learning library
- **Jupyter Notebook**: Interactive computing environment

## Contact Information

For any inquiries or issues, please contact:
- **Pritesh Gandhi**
- **Email**: pgandhi1412@gmail.com
- **GitHub**: [GitHubProfile](https://github.com/Devil-Code)
