# Word Vectors Project

This project implements a word vector representation model using `gensim` and `scikit-learn`. The project leverages word embeddings to visualize semantic relationships between words through dimensionality reduction techniques like PCA and t-SNE. Word categories are represented and plotted to understand contextual similarities.

---

## 📌 Features

* **Word Vector Generation:** Creates word embeddings using `gensim`.
* **Dimensionality Reduction:** Applies PCA or t-SNE for visualizing high-dimensional vectors.
* **Contextual Clustering:** Groups semantically similar words together for analysis.
* **Interactive Plotting:** Generates visual plots of word clusters for exploration.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed:

```bash
sudo apt-get update
sudo apt-get install python3
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:

* `scipy < 1.13`
* `scikit-learn`
* `numpy`
* `gensim`
* `nltk`
* `matplotlib`
* `openai`
* `ipykernel`

---

## 📂 Directory Structure

```
.
├── word_vectors.py          # Main logic for word vector generation
├── test_word_vectors.py     # Unit tests for the word vectors
├── cats.txt                 # List of word categories
├── requirements.txt         # List of dependencies
├── test-plot.png            # Test plot for visualization
├── sample_plot.png          # Sample output of word vector clustering
├── real_data_plot.png       # Plot for real dataset
├── real_data_plotSecond.png # Secondary plot for real dataset
```

---

## 💡 Usage

To generate word vectors and visualize the relationships:

```bash
python3 word_vectors.py
```

To run the tests:

```bash
python3 -m unittest test_word_vectors.py
```

---

### Example:

```plaintext
Input Word List: ["gold", "silver", "copper", "investment", "trade"]

Generated Plot: Displays the relative positioning of words based on semantic similarity.
```

---

## 📝 Memory and Plot Examples

The project generates the following visualizations:

* `test-plot.png`: Initial testing plot with example words.
* `sample_plot.png`: Demonstrates category clustering of words.
* `real_data_plot.png`: Visualizes real-world datasets for analysis.
* `real_data_plotSecond.png`: Secondary visualization for deeper analysis.

---

## 🛠️ Error Handling and Debugging

* **File Not Found:** Ensure all data files are placed in the correct directory.
* **Dependencies Missing:** Run `pip install -r requirements.txt` if you encounter import errors.
* **Unicode Errors:** For non-English words, ensure UTF-8 encoding is used.

---

## 🚀 Future Improvements

* [ ] Integrate with OpenAI API for contextual embeddings.
* [ ] Enhance clustering algorithms for better semantic separation.
* [ ] Add interactive plots with `Plotly` or `Bokeh`.

---

## 🤝 Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.
