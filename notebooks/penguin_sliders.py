import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load the classic penguins dataset
    df = sns.load_dataset("penguins").dropna(subset=["bill_length_mm", "flipper_length_mm", "body_mass_g"])

    # Create interactive sliders matching the dataset range
    bill_slider = mo.ui.slider(start=30, stop=60, step=0.5, value=45, label="📐 Bill Length (mm)")
    flipper_slider = mo.ui.slider(start=170, stop=240, step=1, value=200, label="🪶 Flipper Length (mm)")
    mass_slider = mo.ui.slider(start=2500, stop=6500, step=50, value=4200, label="⚖️ Body Mass (g)")

    # Arrange sliders inside a neat vertical layout block
    mo.md(f"""
    ### 🐧 Design Your Custom Penguin
    Adjust the physical characteristics below to see which real species your penguin closest resembles!

    {mo.vstack([bill_slider, flipper_slider, mass_slider])}
    """)

    return bill_slider, df, flipper_slider, mass_slider, mo, np, plt, sns


@app.cell
def _(bill_slider, df, flipper_slider, mass_slider, mo, np):
    # 1. Grab current slider values
    user_features = np.array([bill_slider.value, flipper_slider.value, mass_slider.value])

    # 2. Compute the historical average for each species
    species_stats = df.groupby("species")[["bill_length_mm", "flipper_length_mm", "body_mass_g"]].mean()

    # 3. Calculate Euclidean distance to each species average (using normalized values for accuracy)
    # We divide by the overall standard deviation so body mass grams don't overwhelm millimeter measurements
    stds = df[["bill_length_mm", "flipper_length_mm", "body_mass_g"]].std().values

    distances = {}
    for species, row in species_stats.iterrows():
        species_features = row.values
        # Normalized Euclidean distance
        dist = np.linalg.norm((user_features - species_features) / stds)
        distances[species] = dist

    # Determine the winner (closest distance)
    predicted_species = min(distances, key=distances.get)

    # Format a clean message
    mo.md(f"""
    ### 🔮 Species Prediction: **{predicted_species.upper()}**

    **Similarity Breakdown (Lower distance means a closer match):**
    *   **Adelie Similarity Score:** {distances['Adelie']:.2f}
    *   **Chinstrap Similarity Score:** {distances['Chinstrap']:.2f}
    *   **Gentoo Similarity Score:** {distances['Gentoo']:.2f}
    """)

    return


@app.cell
def _(df, flipper_slider, mass_slider, mo, plt, sns):
    # Create a visual context map showing where the user's penguin sits relative to the flock
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot the real data points as a backdrop
    sns.scatterplot(
        data=df, 
        x="flipper_length_mm", 
        y="body_mass_g", 
        hue="species", 
        alpha=0.4, 
        ax=ax,
        palette={"Adelie": "#FF8C00", "Chinstrap": "#9932CC", "Gentoo": "#008B8B"}
    )

    # Plot a prominent star mapping the user's custom penguin configuration
    ax.scatter(
        flipper_slider.value, 
        mass_slider.value, 
        color="red", 
        marker="*", 
        s=300, 
        edgecolor="black", 
        linewidth=2,
        label="Your Penguin"
    )

    ax.set_title(f"Your Penguin vs. Real Species Distributions", fontsize=12, fontweight="bold")
    ax.set_xlabel("Flipper Length (mm)")
    ax.set_ylabel("Body Mass (g)")
    ax.legend()
    plt.tight_layout()

    # Convert the matplotlib figure directly to an optimized marimo visual block
    mo.as_html(fig)

    return


if __name__ == "__main__":
    app.run()
