import os
from rich.console import Console
from rich.table import Table


corresponding_label_dir_test = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels\segmented_test"
corresponding_label_dir_train = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels\segmented_train"
corresponding_label_dir_val = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels\segmented_val"
root_dir = r"D:\BusXray\scanbus_training\scanbus_training_4_dataset\labels"
stats_dict = {}
for subdir, dirs, files in os.walk(root_dir):
    # Filter subdirectories based on name
    if "segmented" in os.path.basename(subdir):
        print(f"Processing files in {subdir}...")
        # Process files in the subdirectory here
        stats_dict[f"{subdir}"] = {}
        for file in files:
            file_path = os.path.join(subdir, file)
            # Do something with file_path

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Date", style="dim", width=12)
table.add_column("Title")
table.add_column("Production Budget", justify="right")
table.add_column("Box Office", justify="right")
table.add_row(
    "Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
)
table.add_row(
    "May 25, 2018",
    "[red]Solo[/red]: A Star Wars Story",
    "$275,000,000",
    "$393,151,347",
)
table.add_row(
    "Dec 15, 2017",
    "Star Wars Ep. VIII: The Last Jedi",
    "$262,000,000",
    "[bold]$1,332,539,889[/bold]",
)

console.print(table)