## Description of methods

### TfidfVectorizer

Technology [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf ) is an important step when working with text data.  
- **TF (Term Frequency)** – measures how often a word occurs in a document.  
- **IDF (Inverse Document Frequency)** – evaluates the weight of a word based on its prevalence in a set of documents.  

As a result, TfidfVectorizer transforms a collection of text documents into a feature matrix, where each element reflects the importance of a word in the context of the entire corpus of texts.

### PassiveAggressiveClassifier

The passive-aggressive classifier is an online learning algorithm that is great for data streaming and dynamic systems. It is characterized by the following features:
- **Passivity:** If the model classifies the object correctly, there are almost no changes.  
- **Aggressiveness:** In case of an error, the model is adjusted with maximum rigidity, which allows you to quickly adapt to new data.

Thus, the algorithm efficiently works with continuously incoming data, which is especially important for real-time applications.

## The main steps of the project

1. **Data loading and preprocessing:**
- Import a dataset with news.
   - Cleaning and preparation of text data for further processing.

2. **Feature extraction:**
- Convert text to numeric format using `TfidfVectorizer'.

3. **Model training:**
- Initialization and training of the `PassiveAggressiveClassifier` on the training dataset.
   - Fine-tune hyperparameters to achieve maximum accuracy.

4. **Evaluation of the model:**
- Construction of an error matrix (confusion matrix) to evaluate the results of the model.
   - Visualization of the results of the model using graphs and diagrams.

5. **Visualization:**
- Creation of graphs such as error matrix, ROC curves and others to demonstrate the effectiveness of the model to the customer.

## Installation

To work with the project, you will need to install the following libraries:

- Python 3.7+
- scikit-learn
- numpy
- pandas
- matplotlib (or seaborn for easier visualization)

You can install the necessary libraries using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Project launch

1. Clone the repository:
 ```bash
 git clone https://github.com/yourusername/your-project-name.git
```

2. Run the main training script:
 ```bash
python main.py
```

3. During execution, visualizations will be generated, including the confusion matrix, which will either be saved as images or displayed in a GUI depending on the implementation.
