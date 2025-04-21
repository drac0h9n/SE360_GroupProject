from backend import Sample

import flet
from flet import *


def main(page: Page):
    def on_condition_change(e):
        input_condition_opt.visible = not input_condition_all.value
        page.update()

    def on_submit(e):
        m = input_m.value.strip()
        n = input_n.value.strip()
        k = input_k.value.strip()
        j = input_j.value.strip()
        s = input_s.value.strip()
        condition = input_condition_opt.value.strip()

        if input_condition_all.value:
            inputs = {"M": m, "N": n, "K": k, "J": j, "S": s}
        else:
            inputs = {"M": m, "N": n, "K": k, "J": j, "S": s, "Condition": condition}

        missing_inputs = [key for key, value in inputs.items() if value == ""]

        if missing_inputs:
            if sample_result_info.visible is True:
                sample_result_info.visible = False
                page.update()

            missing_input_info.visible = True
            missing_input_info.value = f"Missing Input: {', '.join(missing_inputs)}"
            page.update()
        else:
            invalid_inputs = []
            if m.isdigit() and n.isdigit() and k.isdigit() and j.isdigit() and s.isdigit():
                m_processed = int(m)
                n_processed = int(n)
                k_processed = int(k)
                j_processed = int(j)
                s_processed = int(s)
                if input_condition_all.value:
                    condition_processed = 'all'
                else:
                    if condition.isdigit():
                        condition_processed = int(condition)
                    else:
                        condition_processed = None

                if not 45 <= m_processed <= 54:
                    invalid_inputs.append("M must be in range from 45 to 54\n")
                if not 7 <= n_processed <= 25:
                    invalid_inputs.append("N must be in range from 7 to 25\n")
                if not 4 <= k_processed <= 7:
                    invalid_inputs.append("K must be in range from 4 to 7\n")
                if not s_processed <= j_processed <= k_processed:
                    invalid_inputs.append("J must be in range from S to K\n")
                if not 3 <= s_processed <= 7:
                    invalid_inputs.append("S must be in range from 3 to 7\n")
                if not condition_processed == 'all' and not condition_processed > 0:
                    invalid_inputs.append("Condition must be a positive integer\n")

                if invalid_inputs:
                    if sample_result_info.visible is True:
                        sample_result_info.visible = False
                        page.update()

                    invalid_input_info.value = (f"Invalid input:\n"
                                                f"{''.join(invalid_inputs)}")
                    invalid_input_info.visible = True
                    page.update()
                else:
                    if missing_input_info.visible is True:
                        invalid_input_info.visible = False
                    if invalid_input_info.visible is True:
                        invalid_input_info.visible = False
                    page.update()

                    sample_result_info.visible = True
                    sample_result_info.value = f"Waiting..."
                    page.update()

                    sample = None
                    sample = Sample(m_processed, n_processed, k_processed, j_processed, s_processed, condition_processed)
                    sample.solve()

                    if sample.ans is not None:
                        sample_result_info.value = str(sample.ans)

                    page.update()

            else:
                if not m.isdigit():
                    invalid_inputs.append("M must be an integer\n")
                if not n.isdigit():
                    invalid_inputs.append("N must be an integer\n")
                if not k.isdigit():
                    invalid_inputs.append("K must be an integer\n")
                if not j.isdigit():
                    invalid_inputs.append("J must be an integer\n")
                if not s.isdigit():
                    invalid_inputs.append("S must be an integer\n")
                if not condition.isdigit() and not input_condition_all.value:
                    invalid_inputs.append("Condition must be an integer\n")

                if invalid_inputs:
                    invalid_input_info.value = (f"Invalid input:\n"
                                                f"{''.join(invalid_inputs)}")
                    invalid_input_info.visible = True
                    page.update()

    input_m = TextField(label="M (45≤M≤54)", width=200, height=60)
    input_n = TextField(label="N (7≤N≤25)", width=200, height=60)
    input_k = TextField(label="K (4≤K≤7)", width=200, height=60)
    input_j = TextField(label="J (S≤J≤K)", width=200, height=60)
    input_s = TextField(label="S (3≤S≤7)", width=200, height=60)
    input_condition_all = Checkbox(label="ALL", value=False, width=90, height=60, on_change=on_condition_change)
    input_condition_opt = TextField(label="Y", visible=True, width=100, height=60)
    submit_button = ElevatedButton(text="Submit", width=100, height=40, on_click=on_submit)

    missing_input_info = Text(visible=False, height=40, size=20)
    invalid_input_info = Text(visible=False, size=20)

    sample_result_info = Text(visible=False, height=40, size=20)

    input_row = Column(
        controls=[
            input_m,
            input_n,
            input_k,
            input_j,
            input_s
        ],
        alignment=alignment.top_left,
        spacing=10
    )
    condition_row = Row(
        controls=[
            input_condition_all,
            input_condition_opt
        ],
        alignment=alignment.top_left,
        spacing=10
    )
    submit_row = Row(
        controls=[
            submit_button,
        ],
        alignment=alignment.top_left,
        spacing=10
    )
    info_row = Column(
        controls=[
            missing_input_info,
            invalid_input_info,
            sample_result_info
        ],
        alignment=alignment.top_left,
        spacing=10
    )

    page.add(
        Column(
            controls=[
                input_row,
                condition_row,
                submit_row,
                info_row
            ],
            run_spacing=10
        )
    )


if __name__ == "__main__":
    flet.app(target=main)
