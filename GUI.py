import sys
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
        elif output_var:  # Otherwise, assume it's text
            output_var.set(result)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {algorithm_module}: {e}")

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

# Function to make prediction for Tab 1
def tab1_function():
    try:
        # Convert inputs
        mass = float(tab1_mass_entry.get())
        year = int(tab1_year_entry.get())
        lat = float(tab1_lat_entry.get())
        long = float(tab1_long_entry.get())

        # Input data for prediction
        input_data = [mass, year, lat, long]

        # Import and run prediction
        from algorithms.classification import predict
        prediction, confidence = predict(input_data)

        # Display result
        output_text1.set(f"Prediction: {prediction}\nConfidence: {confidence:.2%}")
    except ValueError as e:
        output_text1.set(f"Error: {e}")
    except Exception as e:
        output_text1.set(f"An unexpected error occurred: {e}")

# Run prediction button
tab1_button = ttk.Button(tab1_left_frame, text="Run Prediction", command=tab1_function)
tab1_button.pack(pady=10)

# Output frame
tab1_output_frame = tk.LabelFrame(tab1, text="Output Pane", width=400, height=500)
tab1_output_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

output_text1 = tk.StringVar()
output_label1 = tk.Label(tab1_output_frame, textvariable=output_text1, anchor="nw", justify="left")
output_label1.pack(fill="both", expand=True, padx=5, pady=5)

# Tab 2 - Regression with World Map Output
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Regression - Approximate Location")

# Left frame for input
tab2_left_frame = tk.Frame(tab2, width=400)
tab2_left_frame.pack(side="left", fill="y", padx=10, pady=10)

tab2_side_pane = tk.LabelFrame(tab2_left_frame, text="Input Meteorite Features", width=400, height=300)
tab2_side_pane.pack(fill="x", padx=5, pady=5)

# Input fields
tk.Label(tab2_side_pane, text="Mass (g):").pack(anchor="w", padx=5, pady=2)
tab2_mass_entry = tk.Entry(tab2_side_pane)
tab2_mass_entry.pack(fill="x", padx=5, pady=2)

tk.Label(tab2_side_pane, text="Year:").pack(anchor="w", padx=5, pady=2)
tab2_year_entry = tk.Entry(tab2_side_pane)
tab2_year_entry.pack(fill="x", padx=5, pady=2)

# Function to predict and display map for Tab 2
def tab2_function():
    try:
        # Convert inputs
        mass = float(tab2_mass_entry.get())
        year = int(tab2_year_entry.get())

        # Input data for prediction
        input_data = [mass, year]

        # Import and run prediction
        from algorithms.regression1 import predict
        predicted_lat, predicted_long = predict(input_data)

        # Clear previous content in output frame
        for widget in tab2_output_frame.winfo_children():
            widget.destroy()

        # Plot the predicted location on a world map
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap

        fig, ax = plt.subplots(figsize=(8, 6))
        m = Basemap(projection="mill", llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution="c", ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        x, y = m(predicted_long, predicted_lat)
        m.scatter(x, y, c="red", s=500, label="Predicted Location", edgecolors="black", zorder=5)
        plt.legend()
        plt.title("Predicted Meteorite Fall Location")

        # Embed the map in the output frame
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=tab2_output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    except ValueError as e:
        tk.messagebox.showerror("Input Error", f"Error: {e}")
    except Exception as e:
        tk.messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Run prediction button
tab2_button = ttk.Button(tab2_left_frame, text="Run Prediction", command=tab2_function)
tab2_button.pack(pady=10)

# Output frame
tab2_output_frame = tk.LabelFrame(tab2, text="Output Pane", width=400, height=500)
tab2_output_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)


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
