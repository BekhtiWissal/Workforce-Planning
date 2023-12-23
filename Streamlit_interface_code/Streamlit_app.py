import streamlit as st
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import pulp
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import base64
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns

def style_productivity_dataframe(dataframe):
    # Style for larger DataFrame
    style = dataframe.style.set_properties(**{
        'font-size': '20px',  # larger font size
        'text-align': 'center'    # center text alignment
    })

    # Function to apply color based on productivity values
    def highlight_max_min(s):
        if s.name == 'Productivity':
            is_max = s == s.max()
            is_min = s == s.min()
            return ['background-color: green' if v else 'background-color: red' if w else '' for v, w in zip(is_max, is_min)]
        return ['' for _ in s]

    # Apply the highlight function
    style = style.apply(highlight_max_min)
    return style


# Function to convert the image to a base64 string
def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Set page configuration
st.set_page_config(layout="wide")

# Ensure the image is in the same directory as your Streamlit script
background_image_path = "/Users/badreddinehannaoui/Downloads/background_image.png"  # This should be a relative path

# Get base64 string
background_image_base64 = get_image_as_base64(background_image_path)

# Custom CSS to set the background image with opacity
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{background_image_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.5);  /* Adjust the color and opacity here */
        pointer-events: none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def show_productivity_table(productivity_df):
    # Check if the required columns are in the DataFrame
    required_columns = ['Employee ID', 'Hours worked today']
    if not all(column in productivity_df.columns for column in required_columns):
        st.error(f"Uploaded file must contain these columns: {', '.join(required_columns)}")
        return

    # Calculate productivity and fatigue
    productivity_df['Productivity'] = (productivity_df['Hours worked today'] / 4) * 100
    productivity_df['Fatigue'] = (1 - productivity_df['Productivity'] / 100) * 100

    # Create a result table with employee ID, productivity, and fatigue
    result_table = productivity_df[['Employee ID', 'Productivity', 'Fatigue']]

    # Apply custom styling to the result table
    styled_result_table = style_productivity_dataframe(result_table)

    # Display the styled result table
    st.write("## Productivity and Fatigue Results:")
    st.markdown(styled_result_table.to_html(escape=False), unsafe_allow_html=True)


def style_dataframe(dataframe):
    return dataframe.style.set_properties(
        **{
            'background-color': '#f4f4f2',  # light grey background
            'color': '#0a3142',  # dark blue text
            'border-color': 'white',
            'font-size': '20px',  # larger font size
        }
    ).set_table_styles(
        [
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#26547c'),  # darker blue header
                    ('color', 'white'),  # white header text
                    ('font-size', '24px'),  # even larger font size for headers
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('padding', '10px'),  # more padding, making each cell larger
                ]
            }
        ]
    ).hide_index()  # hide the index to make it cleaner

def run_svm_algorithm(employees_df, tasks_df):
    # Merge employees and tasks based on matching skills
    merged_df = pd.merge(employees_df, tasks_df, left_on='Skills', right_on='Required_Skill')

    # Create dummy variables for categorical features
    tasks_features = merged_df['Assembly_Line'].apply(lambda x: [f"Assembly_Line_{i}" for i in range(x)])
    tasks_features = tasks_features.str.join(',').str.get_dummies(',')

    employee_features = pd.concat([
        merged_df['Available_Days'].apply(lambda x: [f"Day_{day}" for day in x.split(', ')]).str.join(',').str.get_dummies(','),
        merged_df['Available_Shifts'].apply(lambda x: [f"Shift_{shift}" for shift in x.split(', ')]).str.join(',').str.get_dummies(','),
        merged_df['Skills'].str.get_dummies(',')],
        axis=1
    )

    # Combine features for model training
    X = pd.concat([employee_features, tasks_features], axis=1)
    y = merged_df['Employee_ID']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the classifier
    clf = SVC()
    clf.fit(X_train, y_train)

    # Predict on the entire dataset
    all_predictions = clf.predict(X)

    # Assign predictions to the merged DataFrame
    merged_df['Assigned_Employee'] = all_predictions

    # Select unique tasks for each employee
    selected_tasks = pd.DataFrame(columns=merged_df.columns)
    selected_employees = set()

    for _, task_row in merged_df.iterrows():
        if task_row['Employee_ID'] not in selected_employees:
            selected_employees.add(task_row['Employee_ID'])
            selected_tasks = pd.concat([selected_tasks, pd.DataFrame([task_row])], ignore_index=True)

            if len(selected_tasks) == len(employees_df):  # Stop once all employees are assigned
                break

    return selected_tasks[['Task_ID', 'Required_Skill', 'Assembly_Line', 'Employee_ID', 'Assigned_Employee']]

def run_pulp_algorithm(employees_df, tasks_df):
    # Create a linear programming problem
    prob = pulp.LpProblem("TaskAssignment", pulp.LpMaximize)

    # Create a dictionary of pulp variables with keys being tuples of (task_id, emp_id)
    assignment_vars = pulp.LpVariable.dicts(
        "assignment",
        ((task_id, emp_id) for task_id in tasks_df.index for emp_id in employees_df.index),
        cat='Binary'
    )

    # Objective Function: Maximize the number of assignments
    prob += pulp.lpSum([assignment_vars[(task_id, emp_id)] for task_id in tasks_df.index for emp_id in employees_df.index])

    # Constraint: Each task is assigned to exactly one employee
    for task_id in tasks_df.index:
        prob += pulp.lpSum([assignment_vars[(task_id, emp_id)] for emp_id in employees_df.index]) == 1

    # Constraint: Each employee is assigned at most one task
    for emp_id in employees_df.index:
        prob += pulp.lpSum([assignment_vars[(task_id, emp_id)] for task_id in tasks_df.index]) <= 1

    # Constraint: Employees can only be assigned tasks they are skilled and available for
    for task_id in tasks_df.index:
        for emp_id in employees_df.index:
            employee = employees_df.loc[emp_id]
            task = tasks_df.loc[task_id]
            has_skill = task['Required_Skill'] in employee['Skills'].split(', ')
            is_available = task['Assembly_Line'] in map(int, employee['Available_Days'].split(', '))
            if not (has_skill and is_available):
                prob += assignment_vars[(task_id, emp_id)] == 0

    # Solve the problem
    prob.solve()

    # Collect the results
    assignments = []
    for task_id in tasks_df.index:
        for emp_id in employees_df.index:
            if pulp.value(assignment_vars[(task_id, emp_id)]) == 1:
                assignments.append({"Task_ID": tasks_df.loc[task_id, "Task_ID"], 
                                    "Employee_ID": employees_df.loc[emp_id, "Employee_ID"]})

    return pd.DataFrame(assignments)

# Function to run the genetic algorithm
def run_genetic_algorithm(employees_df, tasks_df):
    # Your genetic algorithm code
    employees_data = {
        row['Employee_ID']: {
            'days': list(map(int, row['Available_Days'].split(', '))),
            'shifts': list(map(int, row['Available_Shifts'].split(', '))),
            'skills': row['Skills'].split(', ')
        } for _, row in employees_df.iterrows()
    }

    tasks_data = {
        row['Task_ID']: {
            'required_skill': row['Required_Skill'],
            'assembly_line': row['Assembly_Line']
        } for _, row in tasks_df.iterrows()
    }

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_emp", random.choice, list(employees_data.keys()))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_emp, len(tasks_data))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        score = 0
        for i, task_id in enumerate(tasks_data):
            emp_id = individual[i]
            task = tasks_data[task_id]
            employee = employees_data[emp_id]
            if task['required_skill'] in employee['skills']:
                score += 1
        return score,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    best_solution = hof[0]
    assignment = {task_id: best_solution[i] for i, task_id in enumerate(tasks_data)}

    return pd.DataFrame(list(assignment.items()), columns=['Task_ID', 'Assigned_Employee_ID'])

# Function to load data from uploaded files
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    else:
        return None

# Streamlit UI
st.title("Task Distribution System")

# Algorithm selection
algorithm_option = st.sidebar.selectbox("Choose the algorithm:", ["Genetic Algorithm", "Linear programming","SVM Algorithm"])

# File uploaders
uploaded_employees = st.sidebar.file_uploader("Upload Employee Data", type=['xlsx'])
uploaded_tasks = st.sidebar.file_uploader("Upload Task Data", type=['xlsx'])
uploaded_productivity_file = st.sidebar.file_uploader("Upload Productivity Data", type=['xlsx'])


if st.sidebar.button("Run Algorithm"):
    if uploaded_employees is not None and uploaded_tasks is not None:
        employees_df = load_data(uploaded_employees)
        tasks_df = load_data(uploaded_tasks)

        # Ensure that data is loaded
        if employees_df is not None and tasks_df is not None:
            # Run the selected algorithm
            if algorithm_option == "Genetic Algorithm":
                results = run_genetic_algorithm(employees_df, tasks_df)
            elif algorithm_option == "Linear programming":
                results = run_pulp_algorithm(employees_df, tasks_df)
            elif algorithm_option == "SVM Algorithm":
                results = run_svm_algorithm(employees_df, tasks_df)

            # Display results in a styled table
            st.write("Algorithm Results:")
            st.dataframe(style_dataframe(results), use_container_width=True)  # Apply styling to the dataframe
        else:
            st.error("Failed to load data.")
    else:
        st.warning("Please upload the necessary data.")
    


def calculate_productivity(productivity_df):
    productivity_df['Productivity'] = (productivity_df['Hours worked today'] / 4) * 100
    productivity_df['Fatigue'] = (1 - productivity_df['Productivity'] / 100) * 100
    return productivity_df






def show_statistical_analysis(productivity_df):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust the size as needed

    # Histogram of Productivity with Kernel Density Estimate
    sns.histplot(productivity_df['Productivity'], kde=True, ax=axs[0, 0])
    axs[0, 0].set_title("Histogram of Productivity")

    # Boxplot for Productivity
    sns.boxplot(x=productivity_df['Productivity'], ax=axs[0, 1])
    axs[0, 1].set_title("Boxplot for Productivity")

    # Scatter Plot (no additional variable)
    # Plot productivity against itself to show the distribution of data points
    sns.scatterplot(x=productivity_df['Productivity'], y=productivity_df['Productivity'], ax=axs[1, 0])
    axs[1, 0].set_title("Scatter Plot of Productivity vs Productivity")

    # Pie Chart of Productivity Distribution
    ax = plt.subplot(2, 2, 4)  # Create a new subplot at position 4 (bottom right)
    productivity_categories = ['Low', 'Medium', 'High']
    productivity_bins = [0, 33, 66, 100]
    productivity_df['Productivity Category'] = pd.cut(productivity_df['Productivity'], bins=productivity_bins, labels=productivity_categories)
    productivity_counts = productivity_df['Productivity Category'].value_counts()
    plt.pie(productivity_counts, labels=productivity_categories, autopct='%1.1f%%', startangle=90)
    ax.set_title("Pie Chart of Productivity Distribution")

    # Adjust the layout and display the plots
    plt.tight_layout()
    st.pyplot(fig)

      # Pair Plot (for multidimensional data)
    # This will pair up all the quantitative variables in your dataframe and plot them against each other
    st.subheader("Pair Plot of Quantitative Variables")
    pair_plot_fig = sns.pairplot(productivity_df.select_dtypes(include=[np.number]))
    st.pyplot(pair_plot_fig)



# Add a button for showing statistical results
if st.sidebar.button("Show Statistical Analysis"):
    if uploaded_productivity_file is not None:
        productivity_df = pd.read_excel(uploaded_productivity_file, sheet_name='Productivity and Fatigue')
        productivity_df = calculate_productivity(productivity_df)

        # Perform and show statistical analysis
        show_statistical_analysis(productivity_df)
    else:
        st.warning("Please upload the productivity data first.")
if st.sidebar.button("Show Productivity"):    
     if uploaded_productivity_file is not None:
      try:
        # Read the specific sheet from the uploaded file
        productivity_df = pd.read_excel(uploaded_productivity_file, sheet_name='Productivity and Fatigue')
        show_productivity_table(productivity_df)
      except Exception as e:
        st.error(f"Error reading file: {e}")
