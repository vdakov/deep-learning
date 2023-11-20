import json

def save_latex_table_from_json_activation_loss(json_file):
    with open('json_outputs\\{}.json'.format(json_file), 'r') as file:
        json_data = json.load(file)
    data_types = ['LOSS', 'MEAN_ABSOLUTE_ERROR', 'MEAN_SQUARED_ERROR', 'MEAN_ABSOLUTE_PERCENTAGE_ERROR', 'VAL_LOSS',
                                  'VAL_MEAN_ABSOLUTE_ERROR', 'VAL_MEAN_SQUARED_ERROR', 'VAL_MEAN_ABSOLUTE_PERCENTAGE_ERROR']
    activations = list(json_data.keys())
    metrics = list(json_data[activations[0]].keys())

    with open('latex_tables\\{}.tex'.format(json_file), 'w') as file:
        file.write(r"""\begin{table}[h]
\centering
\begin{tabular}{|c|c|c|c|c|c|c|c|c|}
\hline
\multirow{2}{*}{\textbf{Activation}} & \multicolumn{4}{c|}{\textbf{Training}} & \multicolumn{4}{c|}{\textbf{Validation}} \\ \cline{2-9} 
 & \textbf{Loss} & \textbf{MAE} & \textbf{MSE} & \textbf{MAPE} & \textbf{Loss} & \textbf{MAE} & \textbf{MSE} & \textbf{MAPE} \\ \hline
""")

        for activation in activations:
            file.write(f"\\multirow{{4}}{{*}}{{\\textbf{{{activation.capitalize()}}}}} \n")
            for metric in metrics:
                for data_type in data_types:
                    value = json_data[activation][metric][data_type]
                    file.write(f" & {value:.3f} ")
                file.write(r"\\ ")
            file.write("\\hline\n")

        file.write(r"""\end{tabular}
\caption{Evaluation Results for Different Activation Functions.}
\label{tab:evaluation_results}
\end{table}
""")
        

def save_latex_table_json_optimizer(json_file):
    with open('json_outputs\\{}.json'.format(json_file), 'r') as file:
        data = json.load(file)

    header = "Initial Learning Rate & Decay Rate & LOSS & VAL LOSS  \\\\"
    table_rows = []

    for initial_lr, decay_rates in data.items():
        for decay_rate, metrics in decay_rates.items():
            row = f"{initial_lr} & {decay_rate} & {metrics['LOSS']:.5f} & {metrics['VAL_LOSS']:.5f}  \\\\"
            table_rows.append(row)

    with open('latex_tables\\{}.tex'.format(json_file), 'w') as out_file:

        out_file.write("\\begin{table}[h]\n")
        out_file.write("\\centering\n")
        out_file.write("\\begin{tabular}{|c|c|c|c|}\n")
        out_file.write("\\hline\n")
        out_file.write(header + "\n")
        out_file.write("\\hline\n")
        for row in table_rows:
            out_file.write(row + "\n")
        out_file.write("\\hline\n")
        out_file.write("\\end{tabular}\n")
        out_file.write("\\caption{Your Caption Here}\n")
        out_file.write("\\label{tab:my_label}\n")
        out_file.write("\\end{table}\n")


def create_latex_table_from_json_layer_sizes(json_file):
    # Read JSON file
    with open('json_outputs\\{}.json'.format(json_file), 'r') as f:

        data = json.load(f)

    # Extract values
    training_loss = data['relu']['mean_squared_error']['LOSS']
    validation_loss = data['relu']['mean_squared_error']['VAL_LOSS']

    # Create LaTeX table content
    table_content = (
        "\\begin{table}[h]\n"
        "\centering\n"
        "\\begin{tabular}{|c|c|}\n"
        "\\hline\n"
        "\\textbf{Training Loss} & \\textbf{Validation Loss} \\\\\n"
        "\\hline\n"
        f"{training_loss} & {validation_loss} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n\n"
        "\\end{table}\n"
    )

    # Write content to output .tex file
    with open('latex_tables\\{}.tex'.format(json_file), 'w') as output:
        output.write(table_content)


def create_latex_table_from_json_final_training(json_file_training, json_file_test):
    # Read JSON file
    with open('json_outputs\\{}.json'.format(json_file_training), 'r') as f:
        data_training = json.load(f)
    with open('json_outputs\\{}.json'.format(json_file_test), 'r') as f:
        data_test = json.load(f)
    

    # Extract values
    training_loss = data_training['LOSS']
    test_loss = data_test['LOSS']

    # Create LaTeX table content
    table_content = (
        "\\begin{table}[h]\n"
        "\centering\n"
        "\\begin{tabular}{|c|c|}\n"
        "\\hline\n"
        "\\textbf{Training Loss} & \\textbf{Test Loss} \\\\\n"
        "\\hline\n"
        f"{training_loss} & {test_loss} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n\n"
        "\\caption{Loss on the training and test sets from training on the whole training set. Once again MSE is the loss.}\n"
        "\\label{tab:final_results}\n"
        "\\end{table}\n"
    )

    # Write content to output .tex file
    with open('latex_tables\\{}.tex'.format('final_table'), 'w') as output:
        output.write(table_content)