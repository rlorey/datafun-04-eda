import marimo as mo

__generated_with = "0.24.0"
app = mo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load the penguins dataset
    df = sns.load_dataset("penguins").dropna(subset=["bill_length_mm", "flipper_length_mm", "body_mass_g"])

    # Create interactive sliders matching the dataset range
    bill_slider = mo.ui.slider(start=30, stop=60, step=0.5, value=45, label="📐 Bill Length (mm)")
    flipper_slider = mo.ui.slider(start=170, stop=240, step=1, value=200, label="🪶 Flipper Length (mm)")
    mass_slider = mo.ui.slider(start=2500, stop=6500, step=50, value=4200, label="⚖️ Body Mass (g)")

    # Arrange sliders inside a neat vertical layout block
    layout = mo.md(f"""
    ### 🐧 Design Your Custom Penguin
    Adjust the physical characteristics below to see which real species your penguin closest resembles!

    {mo.vstack([bill_slider, flipper_slider, mass_slider])}
    """)

    # Display the layout element on the screen
    layout

    return bill_slider, df, flipper_slider, layout, mass_slider, mo, np, plt, sns


@app.cell
def _(bill_slider, df, flipper_slider, mass_slider, mo, np):
    # Grab current slider values
    user_features = np.array([bill_slider.value, flipper_slider.value, mass_slider.value])

    # Compute the historical average for each species
    species_stats = df.groupby("species")[["bill_length_mm", "flipper_length_mm", "body_mass_g"]].mean()

    # Calculate Euclidean distance to each species average
    stds = df[["bill_length_mm", "flipper_length_mm", "body_mass_g"]].std().values

    distances = {}
    for species, row in species_stats.iterrows():
        species_features = row.values
        dist = np.linalg.norm((user_features - species_features) / stds)
        distances[species] = dist

    # Determine the winner
    predicted_species = min(distances, key=distances.get)

    # Format a clean message
    output = mo.md(f"""
    ### 🔮 Species Prediction: **{predicted_species.upper()}**

    **Similarity Breakdown (Lower distance means a closer match):**
    *   **Adelie Similarity Score:** {distances['Adelie']:.2f}
    *   **Chinstrap Similarity Score:** {distances['Chinstrap']:.2f}
    *   **Gentoo Similarity Score:** {distances['Gentoo']:.2f}
    """)

    # Display the prediction block on the screen
    output

    return distances, output, predicted_species


@app.cell
def _(df, flipper_slider, mass_slider, mo, plt, sns):
    # Create a visual context map showing where the user's penguin sits relative to the flock
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot the real data points as a backdrop with different colors for each species
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

    # Expose the figure object to marimo cell global output so it renders on screen
    plot_output = mo.as_html(fig)
    plt.close(fig)

    # Display the plot element on the screen
    plot_output

    return (plot_output,)


if __name__ == "__main__":
    app.run()

