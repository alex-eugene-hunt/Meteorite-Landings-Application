import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to dynamically run the algorithm and update the GUI
def run_algorithm(algorithm_module, output_var=None, output_frame=None):
    try:
        # Import the module dynamically
        module = __import__(f"algorithms.{algorithm_module}", fromlist=[""])
        result = module.run()

        if isinstance(result, Figure):  # If the result is a Matplotlib figure
            if output_frame:
                for widget in output_frame.winfo_children():
                    widget.destroy()  # Clear previous content
                canvas = FigureCanvasTkAgg(result, master=output_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(result)
        elif output_var:  # Otherwise, assume it's text
            output_var.set(result)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {algorithm_module}: {e}")

# Load dataset and calculate min/max for the required features
def calculate_feature_ranges():
    df = pd.read_csv("dataset/meteorites.csv")
    feature_ranges = {
        "mass (g)": (df["mass (g)"].min(), df["mass (g)"].max()),
        "year": (df["year"].min(), df["year"].max())
    }
    return feature_ranges

def initialize_empty_map():
    """
    Display an empty world map in the output frame when the GUI starts.
    """
    # Clear previous content in the output frame
    for widget in tab2_output_frame.winfo_children():
        widget.destroy()

    # Create a blank world map using Basemap
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    fig, ax = plt.subplots(figsize=(8, 6))
    m = Basemap(projection="mill", llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution="c", ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    plt.title("Meteorite Prediction Map")

    # Embed the map in the output frame
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=tab2_output_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig)

def on_closing():
    """
    Handle the closing of the GUI window.
    Ensures the application terminates properly.
    """
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        # Clean up any additional resources here if necessary
        plt.close("all")
        root.destroy()  # This ends the mainloop and closes the app
        sys.exit()

# Create the main application window
root = tk.Tk()
root.title("Predictive Analysis and Classification of Meteorite Landings Using Data Mining")
root.geometry("800x500")
# Attach the quit handler
root.protocol("WM_DELETE_WINDOW", on_closing)

# Set the app icon
root.iconbitmap("App_Icon.ico")

# Create a notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Tab 1 - Classification with User Input

# Function to validate latitude and longitude inputs
def validate_lat_long(lat, long):
    """
    Validates latitude and longitude inputs.

    Args:
        lat (float): Latitude value.
        long (float): Longitude value.

    Raises:
        ValueError: If latitude or longitude is out of bounds.
    """
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude must be between -90 and 90. You entered {lat}.")
    if not (-180 <= long <= 180):
        raise ValueError(f"Longitude must be between -180 and 180. You entered {long}.")

tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Classification - Fell or Found")

# Left frame for input
tab1_left_frame = tk.Frame(tab1, width=400)
tab1_left_frame.pack(side="left", fill="y", padx=10, pady=10)

tab1_side_pane = tk.LabelFrame(tab1_left_frame, text="Input Meteorite Features", width=400, height=300)
tab1_side_pane.pack(fill="x", padx=5, pady=5)

# Input fields
tk.Label(tab1_side_pane, text="Mass (g):").pack(anchor="w", padx=5, pady=2)
tab1_mass_entry = tk.Entry(tab1_side_pane)
tab1_mass_entry.pack(fill="x", padx=5, pady=2)

tk.Label(tab1_side_pane, text="Year:").pack(anchor="w", padx=5, pady=2)
tab1_year_entry = tk.Entry(tab1_side_pane)
tab1_year_entry.pack(fill="x", padx=5, pady=2)

tk.Label(tab1_side_pane, text="Latitude (reclat):").pack(anchor="w", padx=5, pady=2)
tab1_lat_entry = tk.Entry(tab1_side_pane)
tab1_lat_entry.pack(fill="x", padx=5, pady=2)

tk.Label(tab1_side_pane, text="Longitude (reclong):").pack(anchor="w", padx=5, pady=2)
tab1_long_entry = tk.Entry(tab1_side_pane)
tab1_long_entry.pack(fill="x", padx=5, pady=2)

# Output frame for prediction results and graphs
tab1_output_frame = tk.LabelFrame(tab1, text="Output Pane", width=400, height=500)
tab1_output_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Prediction text box
output_text1 = tk.StringVar()
prediction_label = tk.Label(
    tab1_output_frame, 
    textvariable=output_text1, 
    anchor="nw", 
    justify="left", 
    bg="white", 
    relief="solid", 
    padx=5, 
    pady=5
)
output_text1.set("Prediction results will appear here.")
prediction_label.pack(fill="x", padx=5, pady=5)

# Graph canvas frame
graph_canvas_frame = tk.Frame(tab1_output_frame)
graph_canvas_frame.pack(fill="both", expand=True)
graph_canvas_frame.grid_rowconfigure(0, weight=1)
graph_canvas_frame.grid_rowconfigure(1, weight=1)
graph_canvas_frame.grid_columnconfigure(0, weight=1)
graph_canvas_frame.grid_columnconfigure(1, weight=1)

# Function to initialize placeholder graphs
def initialize_placeholder_graphs():
    """
    Display placeholder graphs in sub-panes on startup.
    """
    for widget in graph_canvas_frame.winfo_children():
        widget.destroy()

    # Create sub-panes for each graph
    for i in range(4):
        sub_pane = tk.LabelFrame(graph_canvas_frame, text=f"Graph {i+1} - Placeholder", width=200, height=200)
        sub_pane.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="nsew")  # Adjust padding for spacing

        # Placeholder plot
        fig, ax = plt.subplots(figsize=(2.5, 2))  # Smaller figure size
        ax.plot()  # Empty plot
        ax.set_title("Placeholder Graph", fontsize=8)
        fig.tight_layout(pad=1.0)  # Adjust padding inside the figure
        canvas = FigureCanvasTkAgg(fig, master=sub_pane)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

# Function to create graphs dynamically
def create_dynamic_graphs(input_data, prediction, confidence):
    """
    Generate and display graphs that provide insights into the machine learning model and dataset.
    """
    from algorithms.classification import get_model_metrics  # Assume this function provides necessary metrics
    metrics = get_model_metrics()  # Retrieve metrics like feature importance, confidence distribution, etc.

    # Clear previous content in the graph canvas
    for widget in graph_canvas_frame.winfo_children():
        widget.destroy()

    # Graph Titles
    titles = [
        "Mass Distribution",
        "Longitude and Latitude on World Map",
        "Metrics of the Model",
        "Confidence Pie Chart"
    ]

    # Create sub-panes and embed insightful graphs
    for i, title in enumerate(titles):
        sub_pane = tk.LabelFrame(graph_canvas_frame, text=f"Graph {i+1} - {title}", width=200, height=200)
        sub_pane.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="nsew")

        # Create and embed plots
        fig, ax = plt.subplots(figsize=(2.5, 2))  # Match placeholder size

        if i == 0:  # Mass Distribution
            # Ensure mass_values are loaded correctly
            mass_values = metrics["mass_values"]  # List of mass values from the dataset
            input_mass = input_data[0]  # Input mass

            # Remove invalid or non-positive values (logarithmic bins require positive values)
            mass_values = [val for val in mass_values if isinstance(val, (int, float)) and val > 0]

            # Check if mass_values are valid
            if len(mass_values) == 0:
                ax.text(0.5, 0.5, "No valid data available", fontsize=8, ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Mass Distribution", fontsize=8)
            else:
                try:
                    # Use logarithmic bins for positive values
                    bins = np.logspace(np.log10(min(mass_values)), np.log10(max(mass_values)), 30)
                except ValueError:
                    # Fallback to linear bins if logarithmic bins are invalid
                    bins = np.linspace(min(mass_values), max(mass_values), 30)

                # Plot the histogram
                counts, bin_edges, patches = ax.hist(mass_values, bins=bins, color="lightblue", edgecolor="black", alpha=0.7)

                # Calculate quadrant boundaries using bin_edges
                total_bins = len(bin_edges) - 1  # Number of bins is len(bin_edges) - 1
                quadrant_size = total_bins // 4  # Divide into 4 quadrants

                for j in range(4):
                    lower_index = j * quadrant_size
                    upper_index = (j + 1) * quadrant_size
                    if j == 3:  # Include all remaining bins in the last quadrant
                        upper_index = total_bins

                    ax.axvspan(
                        bin_edges[lower_index],
                        bin_edges[upper_index],
                        facecolor="lightblue",
                        alpha=0.2,
                        edgecolor="none",
                    )

                # Add a vertical red line for the input mass
                if min(mass_values) <= input_mass <= max(mass_values):
                    ax.axvline(input_mass, color="red", linestyle="--", linewidth=1.5, label=f"Input Mass: {input_mass}")
                else:
                    ax.text(0.5, 0.9, "Input Mass out of range", fontsize=8, ha="center", va="center", transform=ax.transAxes)

                # Set axis labels and title
                ax.set_xlabel("Mass (g)", fontsize=8)
                ax.set_ylabel("Frequency", fontsize=8)
                ax.set_title("Mass Distribution", fontsize=8)

                # Add a legend
                ax.legend(fontsize=7)

                # Adjust x and y limits for better visualization
                ax.set_xlim([min(bin_edges), max(bin_edges)])
                ax.set_ylim([0, max(counts) + 1])

                # Set logarithmic scale for x-axis if bins were logarithmic
                if np.all(np.diff(bins) > 0):  # Check if bins are logarithmic
                    ax.set_xscale("log")



        elif i == 1:  # Longitude and Latitude on World Map
            from mpl_toolkits.basemap import Basemap
            lat, long = input_data[2], input_data[3]  # Input latitude and longitude
            m = Basemap(ax=ax)
            m.drawcoastlines()
            m.drawcountries()
            ax.axhline(lat, color="red", linestyle="-", label=f"Latitude: {lat}")
            ax.axvline(long, color="red", linestyle="-", label=f"Longitude: {long}")
            ax.set_title("Input Location on World Map", fontsize=8)
            ax.legend(fontsize=7)

        elif i == 2:  # Model Metrics
            metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
            metric_values = [
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1_score", 0)
            ]

            # Ensure metric values are valid and not all zeros
            if all(value == 0 for value in metric_values):
                ax.text(0.5, 0.5, "No valid metrics available", fontsize=8, ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Model Metrics", fontsize=8)
            else:
                # Plot the bars
                ax.bar(metric_names, metric_values, color=["blue", "green", "orange", "purple"], alpha=0.7)

                # Set y-axis limit to 1 for consistent scaling
                ax.set_ylim(0, 1)

                # Annotate bars with their values above the bars
                for idx, value in enumerate(metric_values):
                    ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", fontsize=7, color="black")

                # Adjust x-axis tick labels to a smaller font size
                ax.set_xticks(range(len(metric_names)))
                ax.set_xticklabels(metric_names, fontsize=6)

                # Add labels and title
                ax.set_ylabel("Score", fontsize=10)
                ax.set_title("Model Metrics", fontsize=10)

        elif i == 3:  # Confidence Pie Chart
            other_confidence = 1 - confidence
            if prediction == "Fell":
                labels = [f"Fell ({confidence:.2%})", f"Found ({other_confidence:.2%})"]
                sizes = [confidence, other_confidence]
            else:
                labels = [f"Found ({confidence:.2%})", f"Fell ({other_confidence:.2%})"]
                sizes = [confidence, other_confidence]
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                colors=["skyblue", "yellow"],
                textprops={"fontsize": 7}
            )
            ax.set_title("Confidence Pie Chart", fontsize=8)

        fig.tight_layout(pad=1.0)

        # Render the graph
        canvas = FigureCanvasTkAgg(fig, master=sub_pane)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

# Function to make prediction for Tab 1
def tab1_function():
    try:
        # Convert inputs
        mass = float(tab1_mass_entry.get())
        year = int(tab1_year_entry.get())
        lat = float(tab1_lat_entry.get())
        long = float(tab1_long_entry.get())

        # Validate latitude and longitude
        validate_lat_long(lat, long)

        # Input data for prediction
        input_data = [mass, year, lat, long]

        # Import and run prediction
        from algorithms.classification import predict
        prediction, confidence = predict(input_data)

        # Update prediction text
        output_text1.set(f"The meteorite is predicted to have {prediction.lower()} with a confidence of {confidence:.2%}.")

        # Update graphs
        create_dynamic_graphs(input_data, prediction, confidence)

    except ValueError as e:
        output_text1.set(f"Error: {e}")
    except Exception as e:
        output_text1.set(f"An unexpected error occurred: {e}")

# Run prediction button
tab1_button = ttk.Button(tab1_left_frame, text="Run Prediction", command=tab1_function)
tab1_button.pack(pady=10)

# Initialize with placeholder graphs
initialize_placeholder_graphs()

# Tab 2 - Regression with World Map Output
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Regression - Approximate Location")

# Left frame for input
tab2_left_frame = tk.Frame(tab2, width=400)
tab2_left_frame.pack(side="left", fill="y", padx=10, pady=10)

tab2_side_pane = tk.LabelFrame(tab2_left_frame, text="Input Meteorite Features", width=400, height=300)
tab2_side_pane.pack(fill="x", padx=5, pady=5)

# Feature ranges (dynamically calculated)
feature_ranges = calculate_feature_ranges()

# Function to toggle between input methods
def toggle_input_method():
    if input_method.get() == "Sliders":
        sliders_frame.pack(fill="x", expand=True)
        entry_fields_frame.pack_forget()
    elif input_method.get() == "Input Fields":
        entry_fields_frame.pack(fill="x", expand=True)
        sliders_frame.pack_forget()

# Add radio buttons to toggle input method at the top
input_method = tk.StringVar(value="Sliders")
radio_frame = tk.Frame(tab2_side_pane)
radio_frame.pack(fill="x", padx=5, pady=5)

radio_sliders = ttk.Radiobutton(radio_frame, text="Sliders", variable=input_method, value="Sliders", command=toggle_input_method)
radio_fields = ttk.Radiobutton(radio_frame, text="Input Fields", variable=input_method, value="Input Fields", command=toggle_input_method)

radio_sliders.pack(side="left", padx=5, pady=5)
radio_fields.pack(side="left", padx=5, pady=5)

# Create sliders for Tab 2
def create_slider(frame, label_text, feature_name):
    tk.Label(frame, text=label_text).pack(anchor="w", padx=5, pady=2)
    min_val, max_val = feature_ranges[feature_name]
    slider = tk.Scale(frame, from_=min_val, to=max_val, orient="horizontal", resolution=1)
    slider.pack(fill="x", padx=5, pady=2)
    return slider

# Create entry fields for Tab 2
def create_entry(frame, label_text):
    tk.Label(frame, text=label_text).pack(anchor="w", padx=5, pady=2)
    entry = tk.Entry(frame)
    entry.pack(fill="x", padx=5, pady=2)
    return entry

# Sliders and input fields (initially hidden)
sliders_frame = tk.Frame(tab2_side_pane)
sliders_frame.pack(fill="x", expand=True)
mass_slider = create_slider(sliders_frame, "Mass (g):", "mass (g)")
year_slider = create_slider(sliders_frame, "Year:", "year")

entry_fields_frame = tk.Frame(tab2_side_pane)
entry_fields_frame.pack_forget()
mass_entry = create_entry(entry_fields_frame, "Mass (g):")
year_entry = create_entry(entry_fields_frame, "Year:")

# Function to predict and display map for Tab 2
def tab2_function():
    try:
        # Get inputs based on selected method
        if input_method.get() == "Sliders":
            mass = mass_slider.get()
            year = year_slider.get()
        elif input_method.get() == "Input Fields":
            mass = float(mass_entry.get())
            year = float(year_entry.get())

        # Input data for prediction
        input_data = [mass, year]

        # Import and run prediction
        from algorithms.regression1 import predict
        predicted_lat, predicted_long = predict(input_data)

        # Ensure valid predictions
        if not (-90 <= predicted_lat <= 90 and -180 <= predicted_long <= 180):
            raise ValueError("Predicted latitude or longitude is out of valid map bounds.")

        # Update the latitude and longitude display
        predicted_lat_var.set(f"{predicted_lat:.6f}")
        predicted_long_var.set(f"{predicted_long:.6f}")

        # Clear previous content in output frame
        for widget in tab2_output_frame.winfo_children():
            widget.destroy()

        # Plot the predicted location on a world map
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap

        fig, ax = plt.subplots(figsize=(8, 6))
        m = Basemap(
            projection="mill",
            llcrnrlat=-90,
            urcrnrlat=90,
            llcrnrlon=-180,
            urcrnrlon=180,
            resolution="c",
            ax=ax
        )
        m.drawcoastlines()
        m.drawcountries()
        ax.set_aspect('equal')  # Maintain fixed aspect ratio

        # Plot the predicted location
        x, y = m(predicted_long, predicted_lat)
        m.scatter(x, y, c="red", s=100, alpha=0.6, label="Predicted Location", edgecolors="black", zorder=5)
        ax.axhline(y, color="blue", linestyle="--", label=f"Latitude")
        ax.axvline(x, color="green", linestyle="--", label=f"Longitude")
        # Adjust the legend placement
        legend = plt.legend(
            loc="upper center",  # Place legend in the upper center
            bbox_to_anchor=(0.5, 1.15),  # Offset legend to be slightly above the map, near the title
            fontsize=8,
            ncol=3  # Arrange items in a single row
        )

        plt.title("Predicted Meteorite Location")

        # Embed the map in the output frame
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=tab2_output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    except ValueError as e:
        tk.messagebox.showerror("Input Error", str(e))
    except Exception as e:
        tk.messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Run prediction button
tab2_button = ttk.Button(tab2_left_frame, text="Run Prediction", command=tab2_function)
tab2_button.pack(pady=10)

# Frame for latitude and longitude display
lat_long_frame = tk.LabelFrame(tab2_left_frame, text="Predicted Coordinates", width=400, height=100)
lat_long_frame.pack(fill="x", padx=10, pady=10)  # Place it below the button

# Latitude display
tk.Label(lat_long_frame, text="Predicted Latitude:").pack(anchor="w", padx=5, pady=2)
predicted_lat_var = tk.StringVar(value="N/A")
predicted_lat_entry = tk.Entry(lat_long_frame, textvariable=predicted_lat_var, state="readonly", readonlybackground="white")
predicted_lat_entry.pack(fill="x", padx=5, pady=2)

# Longitude display
tk.Label(lat_long_frame, text="Predicted Longitude:").pack(anchor="w", padx=5, pady=2)
predicted_long_var = tk.StringVar(value="N/A")
predicted_long_entry = tk.Entry(lat_long_frame, textvariable=predicted_long_var, state="readonly", readonlybackground="white")
predicted_long_entry.pack(fill="x", padx=5, pady=2)

# Output frame
tab2_output_frame = tk.LabelFrame(tab2, text="Output Pane", width=400, height=500)
tab2_output_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Initialize the output frame with an empty map
initialize_empty_map()

# Tab 3 - Clustering with Subtabs
tab3 = ttk.Frame(notebook)
notebook.add(tab3, text="Clustering - Group Meteorites Geographically")

tab3_left_frame = tk.Frame(tab3, width=400)
tab3_left_frame.pack(side="left", fill="y", padx=10, pady=10)

tab3_button = ttk.Button(tab3_left_frame, text="Run Clustering", command=lambda: tab3_function())
tab3_button.pack(pady=10)

tab3_output_frame = tk.Frame(tab3, width=400)
tab3_output_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Subtabs for graphs
output_notebook = ttk.Notebook(tab3_output_frame)
output_notebook.pack(expand=True, fill="both")

# Frames for each subtab
subtab1 = ttk.Frame(output_notebook)
subtab2 = ttk.Frame(output_notebook)
subtab3 = ttk.Frame(output_notebook)
output_notebook.add(subtab1, text="Graph 1")
output_notebook.add(subtab2, text="Graph 2")
output_notebook.add(subtab3, text="Graph 3")

def tab3_function():
    try:
        # Import clustering script and generate graphs
        from algorithms.clustering import run
        plots = run()

        # Clear existing widgets in subtabs
        for frame in [subtab1, subtab2, subtab3]:
            for widget in frame.winfo_children():
                widget.destroy()

        # Display graphs in the subtabs
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Graph 1
        canvas1 = FigureCanvasTkAgg(plots[0], master=subtab1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)

        # Graph 2
        canvas2 = FigureCanvasTkAgg(plots[1], master=subtab2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)

        # Graph 3
        canvas3 = FigureCanvasTkAgg(plots[2], master=subtab3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill="both", expand=True)

    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to generate graphs: {e}")

# Tab 4 - Regression to Predict Year
tab4 = ttk.Frame(notebook)
notebook.add(tab4, text="Regression - Predict Year")

# Left frame for input
tab4_left_frame = tk.Frame(tab4, width=400)
tab4_left_frame.pack(side="left", fill="y", padx=10, pady=10)

tab4_side_pane = tk.LabelFrame(tab4_left_frame, text="Input Meteorite Features", width=400, height=300)
tab4_side_pane.pack(fill="x", padx=5, pady=5)

# Input fields
tk.Label(tab4_side_pane, text="Mass (g):").pack(anchor="w", padx=5, pady=2)
tab4_mass_entry = tk.Entry(tab4_side_pane)
tab4_mass_entry.pack(fill="x", padx=5, pady=2)

tk.Label(tab4_side_pane, text="Latitude (reclat):").pack(anchor="w", padx=5, pady=2)
tab4_lat_entry = tk.Entry(tab4_side_pane)
tab4_lat_entry.pack(fill="x", padx=5, pady=2)

tk.Label(tab4_side_pane, text="Longitude (reclong):").pack(anchor="w", padx=5, pady=2)
tab4_long_entry = tk.Entry(tab4_side_pane)
tab4_long_entry.pack(fill="x", padx=5, pady=2)

# Function to predict year
def tab4_function():
    try:
        # Gather inputs
        mass = float(tab4_mass_entry.get())
        lat = float(tab4_lat_entry.get())
        long = float(tab4_long_entry.get())

        # Input data for prediction
        input_data = [mass, lat, long]

        # Import and run prediction
        from algorithms.regression2 import predict
        predicted_year, confidence = predict(input_data)

        # Display result
        output_text4.set(f"Predicted Year: {predicted_year:.0f}\nConfidence: {confidence:.2%}")
    except Exception as e:
        output_text4.set(f"Error: {e}")

# Run prediction button
tab4_button = ttk.Button(tab4_left_frame, text="Run Prediction", command=tab4_function)
tab4_button.pack(pady=10)

# Output frame
tab4_output_frame = tk.LabelFrame(tab4, text="Output Pane", width=400, height=500)
tab4_output_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

output_text4 = tk.StringVar()
output_label4 = tk.Label(tab4_output_frame, textvariable=output_text4, anchor="nw", justify="left")
output_label4.pack(fill="both", expand=True, padx=5, pady=5)

# Run the application
root.mainloop()
