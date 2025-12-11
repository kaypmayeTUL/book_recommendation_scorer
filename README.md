# 📚 Library Book Recommendation Scorer

An interactive web application that helps librarians score and rank book purchase recommendations based on their library's checkout history.

## Features

- 📊 **Smart Scoring**: Analyzes books based on subject similarity, LC classification, and author popularity
- 🎯 **Interactive Filtering**: Search, sort, and filter results in real-time
- 📥 **Easy Upload**: Simply upload two CSV files to get started
- 💾 **Export Results**: Download scored recommendations as CSV or formatted report
- ⚙️ **Customizable Weights**: Adjust scoring factors to match your library's priorities

## How It Works

The app analyzes your library's checkout patterns and scores potential purchases based on:

1. **Subject Similarity (50%)** - How well book topics match your popular collections
2. **Checkout Volume (30%)** - Popularity of books in the same LC classification
3. **Author Popularity (20%)** - Success of author's other works in your library

## Quick Start

### Option 1: Use Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy your fork
4. Share the URL with your team!

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/library-recommendation-scorer.git
cd library-recommendation-scorer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Data Requirements

### Checkouts File (CSV)

Required columns:
- `title` - Book titles
- `author` - Author names
- `checkouts` - Number of times checked out (numeric)
- `lc_classification` - Library of Congress classification (optional but recommended)
- `subjects` - Subject headings separated by semicolons (optional but recommended)

**Example:**
```csv
title,author,checkouts,lc_classification,subjects
"The Great Gatsby","Fitzgerald, F. Scott",45,PS3511.I9,"American Literature; 1920s"
"Introduction to Psychology","Smith, John",127,BF121,"Psychology; Textbooks"
```

### Recommendations File (CSV)

Required columns:
- `title` - Book titles
- `author` - Author names (optional but recommended)
- `lc_classification` - Library of Congress classification (optional but recommended)
- `subjects` - Subject headings (optional but recommended)

**Example:**
```csv
title,author,lc_classification,subjects
"Climate Change and Society","Jones, Mary",QC903,"Climate; Environment; Social Science"
"Modern Urban Planning","Davis, Robert",HT166,"Urban Planning; Cities"
```

## Interpreting Scores

**Likelihood Score (0-100):**
- **70-100**: High priority - Very likely to be popular
- **40-69**: Medium priority - Moderately likely to be popular
- **0-39**: Low priority - Less likely to be popular (based on current checkout patterns)

**Component Scores:**
- **Similarity Score**: How well subjects/topics match your popular books
- **Checkout Volume Score**: Popularity of books in the same LC classification
- **Author Popularity Score**: Track record of author's other works

## Deployment to GitHub/Streamlit Cloud

### Step 1: Create GitHub Repository

1. Create a new repository on GitHub
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository, branch (main), and main file (`app.py`)
5. Click "Deploy!"

Your app will be live at: `https://your-app-name.streamlit.app`

## Configuration

### Adjusting Scoring Weights

The default weights are:
- Subject Similarity: 50%
- LC Classification: 30%
- Author Popularity: 20%

You can adjust these in the app interface before scoring.

## Tips for Best Results

- ✅ Include at least 6-12 months of checkout data
- ✅ Include LC classification and subjects columns for better matching
- ✅ Remove test records or books with zero checkouts
- ✅ Run analysis periodically as your collection grows

## Technical Details

**Built with:**
- [Streamlit](https://streamlit.io/) - Web framework
- [Pandas](https://pandas.pydata.org/) - Data processing
- [NLTK](https://www.nltk.org/) - Natural language processing
- [NumPy](https://numpy.org/) - Numerical computing

**Matching Techniques:**
- Word stemming for subject analysis
- Synonym matching using WordNet
- LC classification prefix matching
- Author name normalization

## Troubleshooting

### "File not found" error
- Verify your CSV files have the correct columns
- Check that column names are exactly as specified (case-sensitive)

### All scores are 0
- Ensure your checkouts file includes `subjects` and/or `lc_classification` columns
- Check that these fields aren't empty

### "Memory Error"
- Your data might be too large for the hosting environment
- Try splitting recommendations into smaller batches

## License

MIT License - feel free to modify and use for your library!

## Support

For issues or questions, please open an issue on GitHub.

---

**Made with ❤️ for librarians**
