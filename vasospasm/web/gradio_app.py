from dataclasses import asdict, dataclass
from math import floor, log10
from typing import Literal

import gradio as gr
import joblib
import pandas as pd

from common.classifier_pipelines import ClfCode
from common.dataset import AneurysmLocationMapper1, process_dataset
from common.web.persist_models import get_model_filename
from vasospasm.config import paths
from vasospasm.const import vasospasm_features
from vasospasm.web.data_validation import validate_data

model_name_map = {
    ClfCode.tree_classifier: "Decision Tree Classifier",
    ClfCode.random_forest: "Random Forest Classifier",
    ClfCode.gbt: "Gradient Boosted Tree (Scikit-learn)",
    ClfCode.xgbt: "Gradient Boosted Tree",
    ClfCode.logistic_regression: "Logistic Regression",
    ClfCode.linear_svc: "Linear Support Vector Classifier",
    ClfCode.nonlinear_svc: "Non-linear Support Vector Classifier",
    ClfCode.knn: "K-Nearest Neighbors Classifier",
}

_predict_all_models = False


def get_model_name_from_code(code):
    return model_name_map[code]


def get_model_code_from_name(name):
    return [code for code, model_name in model_name_map.items() if model_name == name][0]


class _defaults:
    model = get_model_name_from_code(ClfCode.logistic_regression)
    training_aneurysmatic_only = True

    age = 50
    sex = "female"
    fisher = 2
    hunt_hess = 2

    aneurysm_present = True
    cns_infection = False
    evd = False

    aneurysm_diameter = 5
    aneurysm_height = 5
    aneurysm_location = "ACom"
    aneurysm_treatment = "coil"


class _choices:
    model_codes = [
        ClfCode.tree_classifier,
        ClfCode.random_forest,
        # ClfCode.gbt,
        ClfCode.xgbt,
        ClfCode.logistic_regression,
        ClfCode.linear_svc,
        ClfCode.nonlinear_svc,
        ClfCode.knn,
    ]
    model_names = [get_model_name_from_code(code) for code in model_codes]
    sex = ["female", "male"]
    hunt_hess = [1, 2, 3, 4, 5]
    fisher = [1, 2, 3, 4]
    aneurysm_location = ["ACA", "ACom", "AICA", "BA", "ICA", "MCA", "PCA", "PCom", "PICA", "SCA", "VA"]
    aneurysm_treatment = ["clip", "coil"]
    cohort = ["Aneurysmatic Cases (n=345)", "Full Cohort (n=503)"]


@dataclass
class Prediction:
    prediction: bool | None
    probability: float | None
    error: str | None


@dataclass
class PatientRecord:
    age_diagnose: float
    sex: Literal["female", "male"]
    hunt_hess: int
    fisher: int
    cns_infection: bool
    EVD: bool
    aneurysm: bool
    aneurysm_diameter: float
    aneurysm_height: float
    aneurysm_location: str | None
    aneurysm_treatment: str | None


def predict_record(
    model_names: list[str],
    model_trained_aneurysmatic_only: bool,
    record: PatientRecord,
) -> list[Prediction]:
    model_dir = paths.repo / "vasospasm/web/models"
    features = vasospasm_features.copy()
    mapper = AneurysmLocationMapper1
    only_with_aneurysm = False
    encode_one_hot = False
    impute = False

    # Create record from inputs
    record = asdict(record)
    df = pd.DataFrame([record])
    validation = validate_data(df)  # Assuming this function validates data as required
    df, features = process_dataset(df, features, mapper, only_with_aneurysm, encode_one_hot, impute)
    x = df[features]

    predictions = []

    for model_name in model_names:
        model_code = get_model_code_from_name(model_name)
        model_path = model_dir / get_model_filename(
            model_code,
            mapper,
            only_with_aneurysm=model_trained_aneurysmatic_only,
        )
        clf_pipeline = joblib.load(model_path)

        y_pred = clf_pipeline.predict(x)
        proba_true = None
        try:
            probas = clf_pipeline.predict_proba(x)
            proba_true = probas[:, 1]
        except:
            pass

        pred = Prediction(
            prediction=bool(y_pred[0]),
            probability=proba_true[0] if proba_true is not None else None,
            error=None,
        )
        predictions.append(pred)

    return predictions


def update_aneurysm_features(aneurysm_present_val):
    if aneurysm_present_val:
        return {
            aneurysm_location: gr.Radio(value=_defaults.aneurysm_location, visible=True),
            aneurysm_diameter: gr.Slider(value=_defaults.aneurysm_diameter, visible=True),
            aneurysm_height: gr.Slider(value=_defaults.aneurysm_height, visible=True),
            aneurysm_treatment: gr.Radio(value=_defaults.aneurysm_treatment, visible=True),
        }
    else:
        return {
            aneurysm_location: gr.Radio(value=None, visible=False),
            aneurysm_diameter: gr.Slider(visible=False),
            aneurysm_height: gr.Slider(visible=False),
            aneurysm_treatment: gr.Radio(value=None, visible=False),
        }


def get_odds(p: float) -> float:
    if p == 0:
        return 0
    elif p == 1:
        return float("inf")
    return p / (1 - p)


def get_odds_string(odds: float, show_digits: int = 4) -> str:
    """

    :param odds:
    :param show_digits:
    :return:

    >>> get_odds_string(1.2344, 4)
    '1.234:1'
    >>> get_odds_string(0.12344, 4)
    '1.234:10'
    >>> get_odds_string(0.012344, 4)
    '1.234:100'
    """
    if odds <= 1e-10:
        return "0 : 1"
    elif odds >= 1e10:
        return "1 : 0"

    digits = floor(log10(odds)) + 1
    odds = round(odds, show_digits - digits)
    numerator = odds
    denominator = 1

    if digits <= 0:
        factor = 10 ** (abs(digits) + 1)
        numerator *= factor
        denominator *= factor
        numerator = round(numerator, show_digits - 1)

    numerator = round(numerator, show_digits)

    return f"{numerator} : {denominator}"


def get_percentage_string(p: float) -> str:
    return f"{100 * p:.2f} %"


def get_prediciton_string(pred: bool) -> str:
    return "Yes" if pred else "No"


def get_text_from_prediction(prediction: Prediction) -> str:
    lines = []
    if prediction.probability is not None:
        lines.append(f"Estimated Probability: {get_percentage_string(prediction.probability)}")
        odds = get_odds(prediction.probability)
        lines.append(f"Estimated Odds: {get_odds_string(odds, 3)}")
    lines.append(f"Model-based Prediction of Vasospasm: {get_prediciton_string(prediction.prediction)}")
    text = "\n".join(lines)
    return text


def get_df_from_predictions(model_names: list[str], predictions: list[Prediction]) -> pd.DataFrame:
    columns = ["Model", "Probability", "Odds", "Prediction"]
    data = []

    for model_name, record_output in zip(model_names, predictions):
        if record_output.probability is None:
            probability = "N/A"
            odds = "N/A"
        else:
            probability = get_percentage_string(record_output.probability)
            odds = get_odds_string(get_odds(record_output.probability), 3)
        prediction = get_prediciton_string(record_output.prediction)

        space_char = "\u00a0"

        probability = probability.rjust(10).replace(" ", space_char)
        odds = odds.rjust(15).replace(" ", space_char)
        prediction = prediction.rjust(3).replace(" ", space_char)

        data.append((model_name, probability, odds, prediction))

    df = pd.DataFrame(data, columns=columns)
    return df


def click_submit_button(
    model_choice: str,
    age_diagnose: int | float,
    sex: Literal["female", "male"],
    hunt_hess: int,
    fisher: int,
    cns_infection: bool,
    EVD: bool,
    aneurysm: bool,
    aneurysm_diameter: float,
    aneurysm_height: float,
    aneurysm_location: str | None,
    aneurysm_treatment: str | None,
    # model_trained_aneurysmatic_only: bool,
    cohort: str,
):
    record = PatientRecord(
        age_diagnose,
        sex,
        hunt_hess,
        fisher,
        cns_infection,
        EVD,
        aneurysm,
        aneurysm_diameter if aneurysm else float("nan"),
        aneurysm_height if aneurysm else float("nan"),
        aneurysm_location if aneurysm else None,
        aneurysm_treatment if aneurysm else None,
    )
    model_trained_aneurysmatic_only = cohort == _choices.cohort[0]
    model_names = [model_choice]
    predictions = predict_record(model_names, model_trained_aneurysmatic_only, record)
    text = get_text_from_prediction(predictions[0])

    if _predict_all_models:
        model_names = _choices.model_names
        predictions = predict_record(model_names, model_trained_aneurysmatic_only, record)
        df = get_df_from_predictions(model_names, predictions)
        html = dataframe_to_html(df)
    else:
        html = ""

    return {
        output: gr.Textbox(value=text, visible=not _predict_all_models),
        # output_dataframe: gr.DataFrame(value=df, visible=True),
        output_html: gr.HTML(value=html, visible=_predict_all_models),
    }


def click_clear_button():
    return {
        output: gr.Textbox(value="", visible=False),
        # output_dataframe: gr.DataFrame(value=pd.DataFrame(), visible=False),
        output_html: gr.HTML(visible=False),
    }


# def change_aneurysmatic_only_checkbox(aneurysmatic_only_val):
#     if aneurysmatic_only_val:
#         return {
#             aneurysm_present: gr.Checkbox(value=True, interactive=False),
#         }
#     else:
#         return {
#             aneurysm_present: gr.Checkbox(interactive=True),
#         }


def change_cohort_rb(cohort_rb_val):
    only_with_aneurysm = cohort_rb_val == _choices.cohort[0]
    if only_with_aneurysm:
        return {
            aneurysm_present: gr.Checkbox(value=True, interactive=False),
        }
    else:
        return {
            aneurysm_present: gr.Checkbox(interactive=True),
        }


def highlight_row(html, model_name):  # Using gold as the highlight color
    from bs4 import BeautifulSoup

    # Parse the HTML
    soup = BeautifulSoup(html, "html.parser")

    # Find all rows in the table body
    rows = soup.find_all("tr")

    # Loop through each row
    for row in rows:
        cells = row.find_all("td")  # Get all data cells
        if cells:  # Check if it's a data row
            if cells[0].text.strip() == model_name:
                # Apply the style to each cell in the row for more consistent rendering
                for cell in cells:
                    # cell['style'] = f"background-color: {color};"
                    cell["style"] = cell.get("style", "") + "background-color: rgba(255, 215, 0, 0.3);"
                break  # Exit after modifying the first matching row

    return str(soup)


def dataframe_to_html(df: pd.DataFrame) -> str:
    from bs4 import BeautifulSoup

    # Convert the DataFrame to HTML
    html = df.to_html(index=False, border=0)

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(html, "html.parser")

    # Find all <tr> tags in the table body (skip the header)
    rows = soup.find_all("tr")

    align_right_idx = [1, 2, 3]

    # Loop through each row and modify the third <td> element (Salary column)
    for row in rows:
        try:
            # Get all <td> tags
            tds = row.find_all(["td", "th"])
            # Apply right alignment style to the third <td> (index 2)
            for idx in align_right_idx:
                tds[idx]["style"] = "text-align: right;"
            # tds[2]['style'] = "text-align: right;"
        except IndexError:
            # This handles cases with fewer than 3 columns per row
            continue

    html = str(soup)
    html = highlight_row(html, "Logistic Regression")
    return html


with gr.Blocks() as app:
    # Disclaimer Modal
    with gr.Row() as disclaimer_modal:
        # gr.Markdown("<br><br><br><br><br>")
        gr.Column(scale=3)  # Left spacer
        with gr.Column(visible=True, scale=4):
            gr.HTML("<div style='height: 15vh;'></div>")  # Top spacer (15% of viewport height)
            gr.Markdown(
                """
                ## ⚠️ Disclaimer (Research Use Only)
            
                This software and the associated machine learning models are provided **exclusively for research and educational purposes**. They are **not intended for and must not be used for any clinical, diagnostic, therapeutic, or medical decision-making purpose**.

                The models have **not been approved, cleared, or certified** by any regulatory authority, including but not limited to the **EU Medical Device Regulation (MDR)**, **US FDA**, or any other national medical device authority.

                No guarantee is made regarding the accuracy, reliability, or suitability of the outputs. The developers and hosting providers make **no warranties** of any kind, express or implied, and **assume no liability** for any use, misuse, or interpretation of the results.

                Users are solely responsible for ensuring that their use of the software complies with all applicable laws, regulations, institutional review requirements, data protection rules (including **GDPR**), and ethical standards.

                **By using this software, you acknowledge that any decisions or actions taken based on its outputs are entirely at your own risk.**
                """
            )
            accept_button = gr.Button("I Accept", variant="primary")
        gr.Column(scale=3)  # Right spacer

    # Main app content (initially hidden)
    with gr.Column(visible=False) as main_content:
        with gr.Row():
            model = gr.Dropdown(
                choices=_choices.model_names,
                label="Select Model",
                value=_defaults.model,
                interactive=True,
                visible=not _predict_all_models,
            )
            cohort = gr.Radio(
                choices=_choices.cohort, label="Model Training Data", value=_choices.cohort[0], interactive=True
            )
            # training_aneurysmatic_only = gr.Checkbox(
            #     label="Model trained on aneurysmatic cases only",
            #     value=_defaults.training_aneurysmatic_only,
            #     interactive=True,
            # )
        with gr.Row():
            age = gr.Slider(0, 100, label="Age at Diagnosis", value=_defaults.age, interactive=True)
            sex = gr.Radio(choices=_choices.sex, label="Sex", value=_defaults.sex, interactive=True)
        with gr.Row():
            hunt_hess = gr.Radio(
                choices=_choices.hunt_hess, label="Hunt-Hess Grade", value=_defaults.hunt_hess, interactive=True
            )
            fisher = gr.Radio(choices=_choices.fisher, label="Fisher Scale", value=_defaults.fisher, interactive=True)
        with gr.Row():
            aneurysm_present = gr.Checkbox(label="Aneurysmatic", value=_defaults.aneurysm_present, interactive=False)
            cns_infection = gr.Checkbox(label="CNS Infection", value=_defaults.cns_infection, interactive=True)
            evd = gr.Checkbox(label="EVD", value=_defaults.evd, interactive=True)
        with gr.Column():
            aneurysm_location = gr.Radio(
                label="Aneurysm Location",
                choices=_choices.aneurysm_location,
                value=_defaults.aneurysm_location,
                interactive=True,
                visible=True,
            )
            with gr.Row():
                aneurysm_diameter = gr.Slider(
                    0,
                    30,
                    step=0.1,
                    label="Aneurysm Diameter (mm)",
                    value=_defaults.aneurysm_diameter,
                    interactive=True,
                    visible=True,
                )
                aneurysm_height = gr.Slider(
                    0,
                    30,
                    step=0.1,
                    label="Aneurysm Height (mm)",
                    value=_defaults.aneurysm_height,
                    interactive=True,
                    visible=True,
                )
            aneurysm_treatment = gr.Radio(
                choices=_choices.aneurysm_treatment,
                label="Aneurysm Treatment",
                value=_defaults.aneurysm_treatment,
                interactive=True,
                visible=True,
            )
        with gr.Row():
            submit_button = gr.Button(value="Submit", interactive=True)
            clear_button = gr.Button(value="Clear", interactive=True)
        with gr.Row():
            output = gr.Textbox(
                label="Model Prediction for Vasospasm",
                interactive=True,
                show_copy_button=True,
                visible=False,
            )
        # with gr.Row():
        #     output_dataframe = gr.DataFrame(
        #         label="Model Prediction for Vasospasm",
        #         headers=["Model", "Probability", "Odds"],
        #         datatype=["str", "str", "str"],
        #         interactive=False,
        #         visible=False,
        #     )
        with gr.Row():
            output_html = gr.HTML(visible=False)

    # Accept disclaimer handler
    def accept_disclaimer():
        return gr.Column(visible=False), gr.Column(visible=True)

    accept_button.click(accept_disclaimer, outputs=[disclaimer_modal, main_content])

    cohort.change(
        change_cohort_rb,
        [cohort],
        [aneurysm_present],
    )

    # training_aneurysmatic_only.change(
    #     change_aneurysmatic_only_checkbox,
    #     [training_aneurysmatic_only],
    #     [aneurysm_present],
    # )

    aneurysm_present.change(
        update_aneurysm_features,
        [aneurysm_present],
        [aneurysm_location, aneurysm_diameter, aneurysm_height, aneurysm_treatment],
    )

    submit_button.click(
        click_submit_button,
        [
            model,
            age,
            sex,
            hunt_hess,
            fisher,
            cns_infection,
            evd,
            aneurysm_present,
            aneurysm_diameter,
            aneurysm_height,
            aneurysm_location,
            aneurysm_treatment,
            # training_aneurysmatic_only,
            cohort,
        ],
        [
            output,
            # output_dataframe,
            output_html,
        ],
    )

    clear_button.click(
        click_clear_button,
        [],
        [
            output,
            # output_dataframe,
            output_html,
        ],
    )


if __name__ == "__main__":
    app.launch()
