# python_nep
This project visualizes and explains the National Education Policy (NEP) in India using Python. It features interactive graphs and insights to highlight key aspects of the NEP.
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr, skew, kurtosis, norm, binom
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# Load the dataset
file_path = './ResearchInformation3.csv'
data = pd.read_csv(file_path)

# Ensure 'Overall' column exists
if 'Overall' not in data.columns:
    raise ValueError("'Overall' column not found in the dataset!")

# Convert 'Attendance' to numerical values
attendance_mapping = {"80%-100%": 1, "60%-79%": 0.8, "Below 60%": 0.6, "Below 40%": 0.4}
data['Attendance'] = data['Attendance'].map(attendance_mapping)

# Convert 'Preparation' to numerical values
preparation_mapping = {"0-1 Hour": 1, "2-3 Hours": 2, "More than 3 Hours": 3}
data['Preparation'] = data['Preparation'].map(preparation_mapping)

# Convert 'Income' to numerical values
income_mapping = {"Low (Below 15,000)": 1, "Lower middle (15,000-30,000)": 2, "Upper middle (30,000-50,000)": 3, "High (Above 50,000)": 4}
data['Income'] = data['Income'].map(income_mapping)

# Convert 'Gaming' to numerical values
gaming_mapping = {"0-1 Hour": 1, "2-3 Hours": 2, "More than 3 Hours": 3}
data['Gaming'] = data['Gaming'].map(gaming_mapping)

# Convert 'Job' to numerical values
job_mapping = {"No": 0, "Yes": 1}
data['Job'] = data['Job'].map(job_mapping)

# Convert 'Extra' to numerical values
extra_mapping = {"No": 0, "Yes": 1}
data['Extra'] = data['Extra'].map(extra_mapping)

# Descriptive Statistics
mean_overall = data['Overall'].mean()
median_overall = data['Overall'].median()
mode_overall = data['Overall'].mode()[0]
std_overall = data['Overall'].std()
var_overall = data['Overall'].var()

class StatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Analysis")

        self.create_widgets()
        self.plot_histogram()

    def create_widgets(self):
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.graph_label = ttk.Label(self.control_frame, text="Select Graph:")
        self.graph_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.graph_selector = ttk.Combobox(self.control_frame, values=[
            "Histogram", "Boxplot", "Hometown Comparison",
            "Department-Wise Analysis", "Gender Impact",
            "Socioeconomic Background", "Study and Leisure Balance",
            "Attendance Trends", "Job Influence",
            "English Proficiency Effect", "Extracurricular Activities Contribution",
            "Semester-Wise Performance Progression", "Last Semester vs. Overall GPA Comparison",
            "Correlation Heatmap", "Performance Predictor", "Clustering Analysis",
            "Measures of Dispersion", "Skewness and Kurtosis", "Probability Analysis",
            "Random Variables", "Mathematical Expectation"
        ])
        self.graph_selector.pack(side=tk.LEFT, padx=5, pady=5)
        self.graph_selector.bind("<<ComboboxSelected>>", self.update_graph)

        self.stats_button = ttk.Button(self.control_frame, text="Show Descriptive Statistics", command=self.show_stats)
        self.stats_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.filter_button = ttk.Button(self.control_frame, text="Filter Data", command=self.filter_data)
        self.filter_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.predict_button = ttk.Button(self.control_frame, text="Predict Performance", command=self.predict_performance)
        self.predict_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.generate_report_button = ttk.Button(self.control_frame, text="Generate Report", command=self.generate_report)
        self.generate_report_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.upload_button = ttk.Button(self.control_frame, text="Upload Data", command=self.upload_data)
        self.upload_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.help_button = ttk.Button(self.control_frame, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.canvas = None

    def plot_histogram(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Overall'], kde=True, color='skyblue', bins=10, ax=ax)
        ax.set_title("Distribution of Overall GPA")
        ax.set_xlabel("Overall GPA")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        self.update_canvas(fig)

    def plot_boxplot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data['Overall'], color='lightgreen', ax=ax)
        ax.set_title("Boxplot of Overall GPA")
        ax.set_xlabel("Overall GPA")
        self.update_canvas(fig)

    def plot_hometown_comparison(self):
        if 'Hometown' in data.columns and 'Gender' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Hometown', y='Overall', hue='Gender', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by Hometown and Gender")
            ax.set_xlabel("Hometown")
            ax.set_ylabel("Overall GPA")
            ax.legend(title="Gender")
            self.update_canvas(fig)
        else:
            print("Error: 'Hometown' or 'Gender' column not found in the dataset!")

    def plot_department_wise_analysis(self):
        if 'Department' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Department', y='Overall', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by Department")
            ax.set_xlabel("Department")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: 'Department' column not found in the dataset!")

    def plot_gender_impact(self):
        if 'Gender' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Gender', y='Overall', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by Gender")
            ax.set_xlabel("Gender")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: 'Gender' column not found in the dataset!")

    def plot_socioeconomic_background(self):
        if 'Income' in data.columns and 'Hometown' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='Income', y='Overall', hue='Hometown', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by Income and Hometown")
            ax.set_xlabel("Income")
            ax.set_ylabel("Overall GPA")
            ax.legend(title="Hometown")
            self.update_canvas(fig)
        else:
            print("Error: 'Income' or 'Hometown' column not found in the dataset!")

    def plot_study_leisure_balance(self):
        if 'Preparation' in data.columns and 'Gaming' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='Preparation', y='Overall', hue='Gaming', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by Preparation Time and Gaming Hours")
            ax.set_xlabel("Preparation Time")
            ax.set_ylabel("Overall GPA")
            ax.legend(title="Gaming Hours")
            self.update_canvas(fig)
        else:
            print("Error: 'Preparation' or 'Gaming' column not found in the dataset!")

    def plot_attendance_trends(self):
        if 'Attendance' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Attendance', y='Overall', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by Attendance")
            ax.set_xlabel("Attendance")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: 'Attendance' column not found in the dataset!")

    def plot_job_influence(self):
        if 'Job' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Job', y='Overall', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by Part-Time Job")
            ax.set_xlabel("Part-Time Job")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: 'Job' column not found in the dataset!")

    def plot_english_proficiency_effect(self):
        if 'English' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='English', y='Overall', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by English Proficiency")
            ax.set_xlabel("English Proficiency")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: 'English' column not found in the dataset!")

    def plot_extracurricular_activities_contribution(self):
        if 'Extra' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Extra', y='Overall', data=data, palette='pastel', ax=ax)
            ax.set_title("Overall GPA by Extracurricular Activities")
            ax.set_xlabel("Extracurricular Activities")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: 'Extra' column not found in the dataset!")

    def plot_semester_wise_performance_progression(self):
        if 'Semester' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(x='Semester', y='Overall', data=data, marker='o', ax=ax)
            ax.set_title("Semester-Wise Performance Progression")
            ax.set_xlabel("Semester")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: 'Semester' column not found in the dataset!")

    def plot_last_semester_vs_overall_gpa(self):
        if 'Last' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='Last', y='Overall', data=data, ax=ax)
            ax.set_title("Last Semester GPA vs. Overall GPA")
            ax.set_xlabel("Last Semester GPA")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: 'Last' column not found in the dataset!")

    def plot_correlation_heatmap(self):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if 'Overall' in numeric_columns and 'Attendance' in numeric_columns and 'Income' in numeric_columns and 'Preparation' in numeric_columns and 'Gaming' in numeric_columns and 'English' in numeric_columns and 'Extra' in numeric_columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = data[numeric_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Heatmap")
            self.update_canvas(fig)
        else:
            print("Error: Required numeric columns not found in the dataset!")

    def plot_performance_predictor(self):
        if 'Attendance' in data.columns and 'Preparation' in data.columns and 'Income' in data.columns and 'Job' in data.columns:
            # Drop rows with any missing values
            filtered_data = data[['Attendance', 'Preparation', 'Income', 'Job', 'Overall']].dropna()

            X = filtered_data[['Attendance', 'Preparation', 'Income', 'Job']]
            y = filtered_data['Overall']

            if X.empty or y.empty:
                messagebox.showerror("Error", "Insufficient data for prediction.")
                return

            reg_model = LinearRegression()
            reg_model.fit(X, y)
            predictions = reg_model.predict(X)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=predictions, y=y, ax=ax)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
            ax.set_title("Performance Predictor")
            ax.set_xlabel("Predicted GPA")
            ax.set_ylabel("Actual GPA")
            self.update_canvas(fig)

            equation = f"GPA = {reg_model.intercept_} + {reg_model.coef_[0]}*Attendance + {reg_model.coef_[1]}*Preparation + {reg_model.coef_[2]}*Income + {reg_model.coef_[3]}*Job"
            messagebox.showinfo("Regression Equation", equation)
        else:
            print("Error: Required columns not found in the dataset!")

    def plot_clustering_analysis(self):
        if 'Overall' in data.columns and 'Attendance' in data.columns and 'Preparation' in data.columns:
            # Drop rows with any missing values
            X = data[['Overall', 'Attendance', 'Preparation']].dropna()
            if X.empty:
                messagebox.showerror("Error", "Insufficient data for clustering.")
                return

            # Perform clustering
            kmeans = KMeans(n_clusters=3)
            clusters = kmeans.fit_predict(X)

            # Create a new DataFrame for the clustering results
            clustering_results = X.copy()
            clustering_results['Cluster'] = clusters

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='Attendance', y='Overall', hue='Cluster', data=clustering_results, palette='viridis', ax=ax)
            ax.set_title("Clustering Analysis")
            ax.set_xlabel("Attendance")
            ax.set_ylabel("Overall GPA")
            self.update_canvas(fig)
        else:
            print("Error: Required columns not found in the dataset!")

    def plot_measures_of_dispersion(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data[['Overall', 'Attendance', 'Income', 'Preparation', 'Gaming', 'English', 'Extra']], ax=ax)
        ax.set_title("Measures of Dispersion")
        ax.set_ylabel("Values")
        self.update_canvas(fig)

    def plot_skewness_and_kurtosis(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        skewness = data[['Overall', 'Attendance', 'Income', 'Preparation', 'Gaming', 'English', 'Extra']].apply(skew)
        kurtosis_values = data[['Overall', 'Attendance', 'Income', 'Preparation', 'Gaming', 'English', 'Extra']].apply(kurtosis)
        sns.barplot(x=skewness.index, y=skewness.values, ax=ax, label='Skewness')
        sns.barplot(x=kurtosis_values.index, y=kurtosis_values.values, ax=ax, label='Kurtosis')
        ax.set_title("Skewness and Kurtosis")
        ax.set_ylabel("Values")
        ax.legend()
        self.update_canvas(fig)

    def plot_probability_analysis(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Overall'], kde=True, color='skyblue', bins=10, ax=ax)
        ax.set_title("Probability Distribution of Overall GPA")
        ax.set_xlabel("Overall GPA")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        self.update_canvas(fig)

    def plot_random_variables(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(norm.rvs(size=1000), kde=True, color='skyblue', bins=10, ax=ax)
        ax.set_title("Normal Distribution")
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        self.update_canvas(fig)

    def plot_mathematical_expectation(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Overall'], kde=True, color='skyblue', bins=10, ax=ax)
        ax.set_title("Mathematical Expectation of Overall GPA")
        ax.set_xlabel("Overall GPA")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        self.update_canvas(fig)

    def update_graph(self, event):
        selected_graph = self.graph_selector.get()
        if selected_graph == "Histogram":
            self.plot_histogram()
        elif selected_graph == "Boxplot":
            self.plot_boxplot()
        elif selected_graph == "Hometown Comparison":
            self.plot_hometown_comparison()
        elif selected_graph == "Department-Wise Analysis":
            self.plot_department_wise_analysis()
        elif selected_graph == "Gender Impact":
            self.plot_gender_impact()
        elif selected_graph == "Socioeconomic Background":
            self.plot_socioeconomic_background()
        elif selected_graph == "Study and Leisure Balance":
            self.plot_study_leisure_balance()
        elif selected_graph == "Attendance Trends":
            self.plot_attendance_trends()
        elif selected_graph == "Job Influence":
            self.plot_job_influence()
        elif selected_graph == "English Proficiency Effect":
            self.plot_english_proficiency_effect()
        elif selected_graph == "Extracurricular Activities Contribution":
            self.plot_extracurricular_activities_contribution()
        elif selected_graph == "Semester-Wise Performance Progression":
            self.plot_semester_wise_performance_progression()
        elif selected_graph == "Last Semester vs. Overall GPA Comparison":
            self.plot_last_semester_vs_overall_gpa()
        elif selected_graph == "Correlation Heatmap":
            self.plot_correlation_heatmap()
        elif selected_graph == "Performance Predictor":
            self.plot_performance_predictor()
        elif selected_graph == "Clustering Analysis":
            self.plot_clustering_analysis()
        elif selected_graph == "Measures of Dispersion":
            self.plot_measures_of_dispersion()
        elif selected_graph == "Skewness and Kurtosis":
            self.plot_skewness_and_kurtosis()
        elif selected_graph == "Probability Analysis":
            self.plot_probability_analysis()
        elif selected_graph == "Random Variables":
            self.plot_random_variables()
        elif selected_graph == "Mathematical Expectation":
            self.plot_mathematical_expectation()

    def update_canvas(self, fig):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_stats(self):
        stats_text = (f"Descriptive Statistics for Overall GPA:\n"
                      f"Mean: {mean_overall}\n"
                      f"Median: {median_overall}\n"
                      f"Mode: {mode_overall}\n"
                      f"Standard Deviation: {std_overall}\n"
                      f"Variance: {var_overall}")
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Descriptive Statistics")
        stats_label = tk.Label(stats_window, text=stats_text, justify=tk.LEFT)
        stats_label.pack(padx=10, pady=10)

    def filter_data(self):
        filter_window = tk.Toplevel(self.root)
        filter_window.title("Filter Data")

    # Create a frame for the filter options
        filter_frame = ttk.Frame(filter_window)
        filter_frame.pack(padx=10, pady=10)

        # Gender filter
        gender_label = ttk.Label(filter_frame, text="Gender:")
        gender_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        gender_combobox = ttk.Combobox(filter_frame, values=[""] + list(data['Gender'].unique()))
        gender_combobox.grid(row=0, column=1, padx=5, pady=5)
        gender_combobox.current(0)

        # Hometown filter
        hometown_label = ttk.Label(filter_frame, text="Hometown:")
        hometown_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        hometown_combobox = ttk.Combobox(filter_frame, values=[""] + list(data['Hometown'].unique()))
        hometown_combobox.grid(row=1, column=1, padx=5, pady=5)
        hometown_combobox.current(0)

        # Department filter
        department_label = ttk.Label(filter_frame, text="Department:")
        department_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        department_combobox = ttk.Combobox(filter_frame, values=[""] + list(data['Department'].unique()))
        department_combobox.grid(row=2, column=1, padx=5, pady=5)
        department_combobox.current(0)

        # Income filter
        income_label = ttk.Label(filter_frame, text="Income Range:")
        income_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        income_combobox = ttk.Combobox(filter_frame, values=[""] + list(data['Income'].unique()))
        income_combobox.grid(row=3, column=1, padx=5, pady=5)
        income_combobox.current(0)

        # Attendance filter
        attendance_label = ttk.Label(filter_frame, text="Attendance Percentage:")
        attendance_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        attendance_combobox = ttk.Combobox(filter_frame, values=[""] + list(data['Attendance'].unique()))
        attendance_combobox.grid(row=4, column=1, padx=5, pady=5)
        attendance_combobox.current(0)

        # Extracurricular filter
        extra_label = ttk.Label(filter_frame, text="Extracurricular Involvement:")
        extra_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        extra_combobox = ttk.Combobox(filter_frame, values=[""] + list(data['Extra'].unique()))
        extra_combobox.grid(row=5, column=1, padx=5, pady=5)
        extra_combobox.current(0)

        def apply_filters():
            gender = gender_combobox.get()
            hometown = hometown_combobox.get()
            department = department_combobox.get()
            income = income_combobox.get()
            attendance = attendance_combobox.get()
            extra = extra_combobox.get()

            filtered_data = data.copy()
            if gender:
                filtered_data = filtered_data[filtered_data['Gender'] == gender]
            if hometown:
                filtered_data = filtered_data[filtered_data['Hometown'] == hometown]
            if department:
                filtered_data = filtered_data[filtered_data['Department'] == department]
            if income:
                filtered_data = filtered_data[filtered_data['Income'] == income]
            if attendance:
                filtered_data = filtered_data[filtered_data['Attendance'] == attendance]
            if extra:
                filtered_data = filtered_data[filtered_data['Extra'] == extra]

            if filtered_data.empty:
                messagebox.showinfo("Info", "No data matches the selected filters.")
                return

            self.update_data(filtered_data)
            filter_window.destroy()

        apply_button = ttk.Button(filter_window, text="Apply Filters", command=apply_filters)
        apply_button.pack(pady=10)

    def update_data(self, filtered_data):
        global data
        data = filtered_data
        self.plot_histogram()

    def predict_performance(self):
        self.plot_performance_predictor()

    def generate_report(self):
        report_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if report_path:
            c = canvas.Canvas(report_path, pagesize=letter)
            c.drawString(100, 750, "Student Performance Analysis Report")
            c.drawString(100, 730, "Descriptive Statistics:")
            c.drawString(120, 710, f"Mean: {mean_overall}")
            c.drawString(120, 690, f"Median: {median_overall}")
            c.drawString(120, 670, f"Mode: {mode_overall}")
            c.drawString(120, 650, f"Standard Deviation: {std_overall}")
            c.drawString(120, 630, f"Variance: {var_overall}")
            c.save()
            messagebox.showinfo("Report Generated", f"Report saved to {report_path}")

    def upload_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            global data
            data = pd.read_csv(file_path)
            self.plot_histogram()

    def show_help(self):
        help_text = (
            "Welcome to the Student Performance Analysis App!\n\n"
            "This app allows you to analyze student performance data using various visualizations and statistical tools.\n\n"
            "Features:\n"
            "- Interactive Data Filtering: Filter data by parameters like gender, hometown, department, income range, attendance percentage, and extracurricular involvement.\n"
            "- Predictive Analysis: Use Linear Regression to predict a student's GPA based on factors like attendance, preparation hours, income, and part-time job.\n"
            "- Correlation Analysis: Compute and display correlations (Pearson and Spearman) between variables like GPA, attendance, and income.\n"
            "- Customizable Reports: Generate PDF reports of key statistics, selected graphs, and predictive insights for education policy planning.\n"
            "- Comparison Dashboard: Compare performance across key metrics such as GPA, test scores, or attendance between different groups (e.g., gender, departments, income levels).\n"
            "- Time Series Analysis: Visualize performance trends over time (e.g., semester-wise or year-wise GPA progression) with line charts.\n"
            "- Clustering Analysis: Implement clustering (using k-means) to group students based on similar performance metrics.\n"
            "- Recommendations Section: Use insights from the data to provide actionable recommendations for students, teachers, and policymakers to improve performance.\n"
            "- Advanced Graph Options: Add customizable graph settings (e.g., changing colors, chart types, adding annotations, etc.).\n"
            "- Data Insights Panel: Provide a quick summary of high-performing students, departments, or common patterns.\n"
            "- Machine Learning Integration: Add a Logistic Regression model to predict the probability of achieving a certain grade threshold (e.g., passing or distinction).\n"
            "- User-Friendly Help Section: Includes an interactive tutorial for navigating the app, understanding graphs, and using features like filtering and predictive tools.\n"
            "- Dynamic Data Upload: Let users upload their datasets in CSV format to analyze their own student performance data.\n"
        )
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_label = tk.Label(help_window, text=help_text, justify=tk.LEFT)
        help_label.pack(padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = StatApp(root)
    root.mainloop()
