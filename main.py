import math
import random

from backend import Sample
from backend import RunCounter
import flet
from flet import *


def main(page: Page, random_univ, run_counter):
    def on_condition_change(e):
        input_condition_opt.visible = not input_condition_all.value
        page.update()

    def on_random_change(e):
        input_manual_univ.visible = not input_random_univ.value
        universe_input_info.visible = not input_random_univ.value
        page.update()

    def on_submit(e):
        m = input_m.value.strip()
        n = input_n.value.strip()
        k = input_k.value.strip()
        j = input_j.value.strip()
        s = input_s.value.strip()
        timeout = input_timeout.value.strip()
        condition = input_condition_opt.value.strip()

        if input_condition_all.value:
            inputs = {"M": m, "N": n, "K": k, "J": j, "S": s, "Timeout": timeout}
        else:
            inputs = {"M": m, "N": n, "K": k, "J": j, "S": s, "Timeout": timeout, "Condition": condition}

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
            if m.isdecimal() and n.isdecimal() and k.isdecimal() and j.isdecimal() and s.isdecimal() and timeout.isdecimal():
                m_processed = int(m)
                n_processed = int(n)
                k_processed = int(k)
                j_processed = int(j)
                s_processed = int(s)
                timeout_processed = float(timeout)
                if input_condition_all.value:
                    condition_processed = math.comb(j_processed, s_processed)
                else:
                    if condition.isdecimal():
                        condition_processed = int(condition)
                    else:
                        condition_processed = None

                if input_random_univ.value:
                    univ = sorted(random_univ.sample(range(1, m_processed + 1), n_processed))
                else:
                    try:
                        arr = list(map(int, input_manual_univ.value.strip().split()))
                        assert len(arr) == n_processed and len(set(arr)) == n_processed and all(
                            1 <= x <= m_processed for x in arr)
                        univ = sorted(arr)
                        universe_input_info.visible = False
                    except Exception:
                        universe_input_info.visible = True
                        universe_input_info.value = f"Invalid Input. Switch To Random Set\n"
                        univ = sorted(random_univ.sample(range(1, m_processed + 1), n_processed))

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
                if not timeout_processed > 0:
                    invalid_inputs.append("Timeout must be greater than 0\n")
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
                        missing_input_info.visible = False
                    if invalid_input_info.visible is True:
                        invalid_input_info.visible = False
                    if universe_set_info.visible is False:
                        universe_set_info.visible = True

                    universe_set_info.value = f"Universe: {univ}"
                    page.update()

                    sample_result_info.visible = True
                    page.update()

                    try:
                        sample = Sample(m_processed,
                                        n_processed,
                                        k_processed,
                                        j_processed,
                                        s_processed,
                                        condition_processed,
                                        univ,
                                        timeout_processed)
                        try:
                            sample_result_info.visible = True
                            sample_result_info.value = "Calculating..."
                            page.update()

                            run_counter.increment((m, n, k, j, s))
                            run_idx = run_counter.get_count((m, n, k, j, s))

                            sample.run(run_idx)

                            if sample.ans is not None:
                                sample_result_info.value = (f"{str(sample.ans)}\n"
                                                            f"{''.join(sample.sets)}"
                                                            f"Solved By: {sample.result['alg']}\n"
                                                            f"Time Consumed: {sample.result['time']:.2f} sec\n"
                                                            )
                            else:
                                sample_result_info.value = f"No result found."
                        finally:
                            del sample
                    except Exception as e:
                        sample_result_info.value = f"Error: {str(e)}"

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
    input_timeout = TextField(label="ILP Waiting Time", width=200, height=60)
    input_condition_all = Checkbox(label="ALL S SAMPLE", value=True, width=200, height=60,
                                   on_change=on_condition_change)
    input_condition_opt = TextField(label="Least S Sample", visible=False, width=400, height=60)

    input_random_univ = Checkbox(label="RANDOM UNIVERSE", value=True, width=200, height=60,
                                 on_change=on_random_change)
    input_manual_univ = TextField(label="Enter N Numbers (1≤X≤M)", visible=False, width=400, height=60)

    submit_button = ElevatedButton(text="SUBMIT", width=200, height=60, on_click=on_submit)

    missing_input_info = Text(visible=False, height=40, size=20)
    invalid_input_info = Text(visible=False, size=20)
    sample_result_info = Text(visible=False, size=20)
    universe_input_info = Text(visible=False, height=40, size=20)
    universe_set_info = Text(visible=False, height=40, size=20)

    input_row_1 = Row(
        controls=[
            input_m,
            input_n
        ],
        alignment=alignment.top_left,
    )
    input_row_2 = Row(
        controls=[
            input_k,
            input_j
        ],
        alignment=alignment.top_left,
    )
    input_row_3 = Row(
        controls=[
            input_s,
            input_timeout
        ],
        alignment=alignment.top_left,
    )
    condition_row = Row(
        controls=[
            input_condition_all,
            input_condition_opt
        ],
        alignment=alignment.top_left,
    )
    universe_row = Row(
        controls=[
            input_random_univ,
            input_manual_univ
        ],
        alignment=alignment.top_left,
    )
    submit_row = Row(
        controls=[
            submit_button,
        ],
        alignment=alignment.top_left,
    )
    info_row = Column(
        controls=[
            missing_input_info,
            invalid_input_info,
            universe_set_info,
            sample_result_info,
            universe_input_info,
        ],
        alignment=alignment.top_left,
        spacing=10
    )
    page_left = Column(
        controls=[
            input_row_1,
            input_row_2,
            input_row_3,
            info_row
        ],
        alignment=alignment.top_left,
    )
    page_right = Column(
        controls=[
            condition_row,
            universe_row,
            submit_row
        ],
        alignment=alignment.top_left,
    )
    page_up = Row(
        controls=[
            page_left,
            page_right
        ],
    )
    page_down = Row(
        controls=[
            info_row
        ]
    )
    scroll_page = Column(
        controls=[
            page_up,
            page_down
        ],
        scroll=flet.ScrollMode.AUTO,
    )
    page.add(
        Container(
            content=scroll_page,
            expand=True,
            padding=10,
        )
    )


if __name__ == "__main__":
    random.seed(0)
    counter = RunCounter()
    flet.app(target=lambda page: main(page, random, counter))
