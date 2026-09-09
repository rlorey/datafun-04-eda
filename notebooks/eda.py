import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dr. Case Notebook
    # Exploratory Data Analysis (EDA)

    - Author: [Denise Case](https://github.com/denisecase/)
    - Repository: [datafun-04-eda](https://github.com/denisecase/datafun-04-eda/)
    - Purpose: Demonstrate a reliable, repeatable EDA process for a new tabular dataset
    - Dataset: Palmer Penguins
    - Source: [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/)
    - There is a docs/data-card.md has more information
    - Date: 2026-08

    ## Goal

    Exploratory Data Analysis helps us understand a new dataset before deeper analysis. This notebook models a relatively standard process for initial EDA:

    1. LOAD the data
    2. INSPECT structure and grain
    3. CHECK data quality
    4. CLASSIFY VARIABLES (numeric, categorical) for analysis
    5. VISUALIZE numeric and categorical distributions
    6. EXPLORE numeric relationships
        - correlation matrix
        - heatmap of all possible x vs y relationshiops
        - custom selected pair correlation
        - custom selected pair scatter plot
    7. SUMMARIZE narrate findings and next questions

    The mechanics of plotting basic charts are delegated to `eda-vizkit`.
    Analysts should focus on what the data means and what warrants investigation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 1. Project Setup and Imports

    All imports and configuration appear once near the top.

    `eda-vizkit` provides reusable visualization functions that return Matplotlib `Axes` objects.
    The analyst works here to choose what to look at and how to interpret the findings.
    """)
    return


@app.cell
def _():
    import logging
    from typing import Final

    from datafun_toolkit.logger import get_logger, log_header
    from eda_vizkit import (
        show_categorical_distribution,
        show_missing_values,
        show_numeric_by_category,
        show_numeric_distribution,
        show_numeric_relationship,
    )
    from IPython.display import display  # Needed for ty
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    LOG: logging.Logger = get_logger("EDA-NB", level="DEBUG")
    log_header(LOG, "EDA-NB")

    DATASET_NAME: Final[str] = "penguins"

    # Look at the data to determine numeric columns
    NUMERIC_COLS: Final[list[str]] = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    # Look at the data to determine categorical columns
    CATEGORICAL_COLS: Final[list[str]] = [
        "species",
        "island",
        "sex",
    ]

    # Pandas display configuration (helps in notebooks)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 120)

    LOG.info("Imports and configuration complete.")
    return (
        CATEGORICAL_COLS,
        DATASET_NAME,
        LOG,
        NUMERIC_COLS,
        display,
        pd,
        plt,
        show_categorical_distribution,
        show_missing_values,
        show_numeric_by_category,
        show_numeric_distribution,
        show_numeric_relationship,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 2. Load the Data

    Load a dataset into a DataFrame and confirm.

    This project uses Seaborn's built-in Palmer Penguins dataset.
    Other EDA projects may load data from CSV, JSON, databases, APIs, or other sources.
    """)
    return


@app.cell
def _(DATASET_NAME: "Final[str]", LOG: "logging.Logger", pd, sns):
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = sns.load_dataset(DATASET_NAME)

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.info(df.head())
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 3. Establish Grain and Inspect Structure

    Before analyzing individual variables, establish the **grain**.

    **Grain answers: What does one row represent?**

    For this dataset:

    > One row represents one observed penguin.

    Determining the dataset structure includes:

    - number of observations (data rows)
    - number of variables (colums)
    - column names
    - storage data types
    - non-null counts

    Storage data type indicates the type of value stored.
    For example, a year may be stored as an integer but still function as an ordered study label (category).
    """)
    return


@app.cell
def _(LOG: "logging.Logger", df: "pd.DataFrame"):
    # df.shape attribute provides a tuple of (number of rows, number of columns)
    num_rows, num_cols = df.shape

    LOG.info(f"Dataset shape: {num_rows} rows, {num_cols} columns")
    LOG.info(f"Columns: {list(df.columns)}")

    # Display basic information about the DataFrame
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 4. Create Data Dictionary and Inspect Data Quality

    WHY: A data dictionary helps with understanding the structure and quality of the data.

    WHY: Missing data is normal. We need to find out how much and where.

    Missing data affects:

    - Visualizations
    - Summary statistics
    - Machine learning models

    LOOK FOR:

    - missing values
    - duplicate rows
    - unusual or unexpected values

    The visualization shows evidence.
    It does **not** decide whether missing data is acceptable or what should be done about it - that's for the analyst to decide.
    """)
    return


@app.cell
def _(LOG: "logging.Logger", df: "pd.DataFrame", pd):
    data_dictionary = pd.DataFrame({'column': df.columns, 'dtype': [str(dtype) for dtype in df.dtypes], 'missing_count': df.isna().sum().values, 'missing_pct': (df.isna().mean() * 100).round(2).values, 'unique_count': [df[_column].nunique(dropna=True) for _column in df.columns]})
    LOG.info(f'Data dictionary:\n{data_dictionary}')
    duplicate_count = int(df.duplicated().sum())
    LOG.info(f'Duplicate rows detected: {duplicate_count}')
    return


@app.cell
def _(df: "pd.DataFrame", plt, show_missing_values):
    _ax = show_missing_values(df)
    _ax.set_title('Missing Values in Palmer Penguins')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 5. Classify Variables for Analysis

    For this EDA, classify variables by how we intend to analyze them.

    ### Continuous numeric variables

    - `bill_length_mm`
    - `bill_depth_mm`
    - `flipper_length_mm`
    - `body_mass_g`

    ### Categorical variables

    - `species`
    - `island`
    - `sex`

    ### Ordered study variable

    - `year`

    The variable classification guides the standard inspection we perform next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 6. Review Numeric Distributions

    A numeric distribution should be inspected before using the variable in relationship analysis.

    For each important continuous numeric variable, review:

    - count
    - missing values
    - mean
    - standard deviation
    - quartiles
    - minimum and maximum
    - histogram

    The summary statistics are computed directly from the data.
    The histogram makes the distribution visually inspectable.
    """)
    return


@app.cell
def _(
    LOG: "logging.Logger",
    NUMERIC_COLS: "Final[list[str]]",
    df: "pd.DataFrame",
):
    numeric_summary = df[NUMERIC_COLS].describe().T
    LOG.info(f"Numeric summary:\n{numeric_summary}")
    return


@app.cell
def _(
    NUMERIC_COLS: "Final[list[str]]",
    df: "pd.DataFrame",
    plt,
    show_numeric_distribution,
):
    for _column in NUMERIC_COLS:
        _ax = show_numeric_distribution(df, column=_column)
        _ax.set_title(f'Distribution of {_column}')
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 7. Review Categorical Distributions

    For each important categorical variable, inspect category frequencies to show:

    - dominant categories
    - rare categories
    - missing values
    - unexpected category labels
    """)
    return


@app.cell
def _(
    CATEGORICAL_COLS: "Final[list[str]]",
    df: "pd.DataFrame",
    display,
    plt,
    show_categorical_distribution,
):
    for _column in CATEGORICAL_COLS:
        counts = df[_column].value_counts(dropna=False)
        display(counts.to_frame(name='count'))
        _ax = show_categorical_distribution(df, column=_column)
        _ax.set_title(f'Distribution of {_column}')
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 8. Investigate a Numeric Relationship

    After inspecting variables individually, choose a relationship worth investigating.

    Question:

    > Do penguins with longer flippers also tend to have greater body mass?

    Both variables are continuous numeric, so a scatter plot is appropriate evidence.
    """)
    return


@app.cell
def _(
    LOG: "logging.Logger",
    df: "pd.DataFrame",
    plt,
    show_numeric_relationship,
):
    relationship_df = df[['flipper_length_mm', 'body_mass_g']].dropna()
    correlation = relationship_df['flipper_length_mm'].corr(relationship_df['body_mass_g'])
    LOG.info(f'Correlation between flipper_length_mm and body_mass_g: {correlation:.3f}')
    _ax = show_numeric_relationship(df, x='flipper_length_mm', y='body_mass_g')
    _ax.set_title('Flipper Length vs. Body Mass')
    _ax.set_xlabel('Flipper Length (mm)')
    _ax.set_ylabel('Body Mass (g)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finding

    Flipper length and body mass show a strong positive relationship in this dataset.

    The correlation and scatter plot provide evidence for that finding.

    A finding is a human conclusion supported by analytical evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 9. Compare a Numeric Variable Across Categories

    Question:

    > How does flipper length differ across penguin species?

    This combines:

    - one categorical variable: `species`
    - one continuous numeric variable: `flipper_length_mm`

    A grouped distribution view helps compare the species.
    """)
    return


@app.cell
def _(df: "pd.DataFrame", display, plt, show_numeric_by_category):
    group_summary = df.groupby('species', observed=True)['flipper_length_mm'].describe()
    display(group_summary)
    _ax = show_numeric_by_category(df, numeric='flipper_length_mm', category='species')
    _ax.set_title('Flipper Length by Species')
    _ax.set_xlabel('Species')
    _ax.set_ylabel('Flipper Length (mm)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finding

    Flipper-length distributions differ across penguin species.

    This suggests that species is relevant when interpreting penguin measurements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 10. Summary and Next Questions

    Instructional notes: A repeatable EDA process offers:

    - a clear understanding of the dataset grain
    - an inventory of variables
    - basic data-quality evidence
    - standard distribution evidence
    - selected relationship evidence
    - findings supported by that evidence
    - new questions worth investigating

    At the end, of your notebook, provide:

    -  brief summary of your findings
    -  suggested next steps

    WHY: EDA is not a final report.
    The summary of your data exploration captures you found
    and what you would like to investigate next.

    Your summary will typically be provided in Markdown
    as it is good for narrative.

    ### Analyst-Provided Custom Findings

    - The dataset contains missing values in several variables.
    - The continuous measurements have different ranges and distributions.
    - Flipper length and body mass have a strong positive relationship.
    - Flipper-length distributions differ across species.

    ### Analyst-Provided Custom Suggested Next Steps

    - Why do the species differ in flipper length?
    - Does the relationship between flipper length and body mass differ by species?
    - Which missing values matter for a specific future analysis?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reminder: Run All before sending to GitHub

    Before saving the notebook and pushing to GitHub, use **Run All**
    so outputs are generated in the correct order.
    """)
    return


if __name__ == "__main__":
    app.run()
