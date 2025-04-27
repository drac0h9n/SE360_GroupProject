# main.py
# 使用 Flet 构建图形用户界面 (GUI)，处理用户输入，调用 backend 计算，并显示结果。
# 当 K=6 时，会将结果保存到 SQLite 数据库。
# 包含手动输入 Universe 和手动指定覆盖条件 y 的选项。

import flet as ft
import random
import time
import json # 用于序列化列表以便存入数据库
import sys # 用于检查命令行参数
import os # 用于路径操作

# 导入后端逻辑和数据库操作
from backend import Sample, RunCounter, comb # 导入 Sample 类, RunCounter 类, 和 comb 函数
import db # 导入数据库操作模块 (db.py)

# --- 全局变量 ---
g_counter = RunCounter()
# random_seed = 0 # 固定种子，用于调试和比较
random_seed = int(time.time()) # 使用当前时间作为种子，每次运行不同
random.seed(random_seed)
print(f"使用的随机种子: {random_seed}")

# --- 主应用函数 ---
def main(page: ft.Page):
    page.title = "组合覆盖问题求解器"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 800
    page.window_height = 750 # 稍微增加高度以容纳新选项

    # --- 输入控件 ---
    txt_m = ft.TextField(label="M (基础集合大小)", hint_text="例如: 45", width=180, value="45")
    txt_n = ft.TextField(label="N (Universe 大小)", hint_text="例如: 8", width=180, value="8")
    txt_k = ft.TextField(label="K (覆盖块大小)", hint_text="例如: 6", width=180, value="6")
    txt_j = ft.TextField(label="J (子集大小)", hint_text="例如: 4", width=180, value="4")
    txt_s = ft.TextField(label="S (最小交集)", hint_text="例如: 4", width=180, value="4")
    txt_timeout = ft.TextField(label="超时时间 (秒)", hint_text="例如: 60", width=180, value="60")

    # --- Universe 输入选项 ---
    chk_manual_univ = ft.Checkbox(label="手动输入 Universe", value=False, on_change=None) # on_change 稍后定义
    # 手动输入 Universe 的文本框，初始隐藏
    txt_manual_univ = ft.TextField(
        label="输入 N 个数字 (空格分隔, 1 到 M 之间)",
        visible=False,
        width=400,
        hint_text="例如: 1 5 10 15 20 25 30 35"
    )

    # --- 覆盖条件 y 输入选项 ---
    # y = C(k,s) * C(n-k, j-s) 是旧y； y = C(j,s) 是新y
    chk_specify_y = ft.Checkbox(label="手动指定 y (覆盖条件)", value=False, on_change=None) # on_change 稍后定义
    # 手动输入 y 的文本框，初始隐藏
    txt_specify_y = ft.TextField(label="输入 y 值 (1 到 C(j,s))", visible=False, width=250, hint_text="输入一个正整数")

    # --- 输出控件 ---
    # 用于显示理论上的覆盖度 y = C(k,s)*C(n-k,j-s)
    theoretical_y_info = ft.Text("理论最大覆盖度 (每个j集能被覆盖的总次数 C(k,s)*C(n-k,j-s)): 等待输入...", size=12)
    # 用于显示每个j集需要的最小覆盖次数 C(j,s)
    max_single_j_coverage = ft.Text("每个j集最大内部覆盖可能 (C(j,s)): 等待输入...", size=12)
    # 用于显示计算结果
    sample_result_info = ft.Text("计算结果将显示在这里...", size=12, selectable=True)
    # 用于显示日志信息或错误
    log_output = ft.ListView(expand=True, spacing=5, auto_scroll=True)

    # --- 按钮 ---
    submit_button = ft.ElevatedButton(text="开始计算", on_click=None)
    clear_log_button = ft.ElevatedButton(text="清空日志", on_click=lambda _: clear_log())

    # --- 日志函数 ---
    def log_message(message: str):
        """向日志区域添加一条带时间戳的消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_output.controls.append(ft.Text(f"[{timestamp}] {message}", size=11, selectable=True))
        page.update()

    def clear_log():
        """清空日志区域"""
        log_output.controls.clear()
        log_message("日志已清空。")
        page.update()

    # --- 显示信息/错误 的辅助函数 ---
    def show_info_message(text_control: ft.Text, message: str, is_error: bool = False):
        """更新指定的 Text 控件来显示信息或错误"""
        text_control.value = message
        text_control.color = ft.Colors.RED if is_error else ft.Colors.BLACK
        page.update()

    # --- 输入验证和处理 ---
    def validate_and_get_int(field: ft.TextField, name: str, min_val: int = 0) -> int | None:
        """验证输入是否为整数，并返回整数值，否则返回 None 并记录日志"""
        try:
            value = int(field.value)
            if value < min_val:
                log_message(f"错误: {name} ({value}) 不能小于 {min_val}。")
                show_info_message(sample_result_info, f"输入错误: {name} 不能小于 {min_val}。", is_error=True)
                return None
            return value
        except ValueError:
            log_message(f"错误: {name} ('{field.value}') 必须是一个有效的整数。")
            show_info_message(sample_result_info, f"输入错误: {name} 必须是一个整数。", is_error=True)
            return None

    # --- 动态更新 y 相关信息 ---
    def update_y_related_info(e=None): # 接受一个可选的事件参数
        """当 N, K, J, S 输入框的值改变时，计算并更新理论 y 值和 C(j,s)"""
        n = validate_and_get_int(txt_n, "N", 1)
        k = validate_and_get_int(txt_k, "K", 1)
        j = validate_and_get_int(txt_j, "J", 1)
        s = validate_and_get_int(txt_s, "S", 0) # S 可以是 0

        # 更新 C(j,s) 显示
        if j is not None and s is not None:
            try:
                if j < s:
                    max_single_cov_val = 0
                else:
                    max_single_cov_val = comb(j, s)
                max_single_j_coverage.value = f"单j集最大内部覆盖可能 (C(j,s)): {max_single_cov_val}"
                max_single_j_coverage.color = ft.Colors.BLACK if j >= s else ft.Colors.ORANGE
                # 更新手动 y 输入框的标签
                txt_specify_y.label = f"输入 y 值 (1 到 {max_single_cov_val})" if max_single_cov_val > 0 else "输入 y 值 (无效)"
                txt_specify_y.update()
            except ValueError as combo_err:
                 max_single_j_coverage.value = f"单j集最大内部覆盖可能 (C(j,s)): 计算错误 ({combo_err})"
                 max_single_j_coverage.color = ft.Colors.RED
                 txt_specify_y.label = "输入 y 值 (需有效J,S)"
                 txt_specify_y.update()
            except Exception as calc_err:
                 max_single_j_coverage.value = f"单j集最大内部覆盖可能 (C(j,s)): 未知错误 ({calc_err})"
                 max_single_j_coverage.color = ft.Colors.RED
                 txt_specify_y.label = "输入 y 值 (计算错误)"
                 txt_specify_y.update()
        else:
             max_single_j_coverage.value = "单j集最大内部覆盖可能 (C(j,s)): 等待有效 J, S..."
             max_single_j_coverage.color = ft.Colors.GREY
             txt_specify_y.label = "输入 y 值" # 恢复默认标签
             txt_specify_y.update()

        # 更新理论 y = C(k,s)*C(n-k,j-s) 显示
        if all(v is not None for v in [n, k, j, s]):
            try:
                if not (0 <= s <= k <= n and 0 <= s <= j <= n):
                    theoretical_y_info.value = "理论覆盖度 (C(k,s)C(n-k,j-s)): 参数无效"
                    theoretical_y_info.color = ft.Colors.ORANGE
                elif k < s or n - k < j - s:
                     theoretical_y_value = 0
                     theoretical_y_info.value = f"理论覆盖度 (C(k,s)C(n-k,j-s)): {theoretical_y_value} (组合参数无效)"
                     theoretical_y_info.color = ft.Colors.BLACK
                else:
                    theoretical_y_value = comb(k, s) * comb(n - k, j - s)
                    theoretical_y_info.value = f"理论覆盖度 (C(k,s)C(n-k,j-s)): {theoretical_y_value}"
                    theoretical_y_info.color = ft.Colors.BLACK
            except ValueError as combo_error:
                 theoretical_y_info.value = f"理论覆盖度 (C(k,s)C(n-k,j-s)): 计算错误 ({combo_error})"
                 theoretical_y_info.color = ft.Colors.RED
            except Exception as calc_error:
                 theoretical_y_info.value = f"理论覆盖度 (C(k,s)C(n-k,j-s)): 未知错误 ({calc_error})"
                 theoretical_y_info.color = ft.Colors.RED
        else:
            theoretical_y_info.value = "理论覆盖度 (C(k,s)C(n-k,j-s)): 等待有效输入 N, K, J, S..."
            theoretical_y_info.color = ft.Colors.GREY

        page.update()

    # 绑定事件到输入框
    txt_n.on_change = update_y_related_info
    txt_k.on_change = update_y_related_info
    txt_j.on_change = update_y_related_info
    txt_s.on_change = update_y_related_info

    # --- Checkbox 事件处理 ---
    def on_manual_univ_change(e):
        """控制手动 Universe 输入框的显隐"""
        is_manual = chk_manual_univ.value
        txt_manual_univ.visible = is_manual
        if not is_manual:
            txt_manual_univ.value = "" # 清空输入
            txt_manual_univ.error_text = None # 清除错误提示
        page.update()

    def on_specify_y_change(e):
        """控制手动 y 输入框的显隐"""
        is_manual = chk_specify_y.value
        txt_specify_y.visible = is_manual
        if not is_manual:
            txt_specify_y.value = "" # 清空输入
            txt_specify_y.error_text = None # 清除错误提示
        # 可以在这里考虑是否改变 theoretical_y_info 的状态，但目前保持显示理论值
        page.update()

    # 绑定 Checkbox 事件
    chk_manual_univ.on_change = on_manual_univ_change
    chk_specify_y.on_change = on_specify_y_change

    # --- 提交按钮的回调函数 ---
    def on_submit(e):
        """处理提交按钮点击事件，执行计算并显示结果"""
        log_message("收到计算请求...")
        show_info_message(sample_result_info, "正在处理输入参数...")
        submit_button.disabled = True # 禁用按钮
        page.update()

        # 重置之前的错误提示
        txt_manual_univ.error_text = None
        txt_specify_y.error_text = None

        error_occurred = False # 标记是否发生错误

        # 1. 获取并验证基本参数
        m_processed = validate_and_get_int(txt_m, "M", 1)
        n_processed = validate_and_get_int(txt_n, "N", 1)
        k_processed = validate_and_get_int(txt_k, "K", 1)
        j_processed = validate_and_get_int(txt_j, "J", 1)
        s_processed = validate_and_get_int(txt_s, "S", 0)
        timeout_val = validate_and_get_int(txt_timeout, "超时时间", 1)

        # 检查基本参数
        if None in [m_processed, n_processed, k_processed, j_processed, s_processed, timeout_val]:
            log_message("基本参数验证失败。")
            error_occurred = True
            # show_info_message 已经在 validate_and_get_int 中调用

        if error_occurred:
            submit_button.disabled = False
            page.update()
            return

        timeout_seconds = timeout_val

        # 2. 进一步检查参数间的逻辑关系
        validation_errors = []
        if n_processed > m_processed:
            validation_errors.append(f"N ({n_processed}) 不能大于 M ({m_processed})。")
        if not (0 <= s_processed <= k_processed <= n_processed and 0 <= s_processed <= j_processed <= n_processed):
             validation_errors.append(f"参数逻辑错误: 需满足 0 <= S({s_processed}) <= K({k_processed}) <= N({n_processed}) "
                                      f"且 0 <= S({s_processed}) <= J({j_processed}) <= N({n_processed})。")
        if validation_errors:
            error_msg = "\n".join(validation_errors)
            log_message(f"参数逻辑错误: {error_msg}")
            show_info_message(sample_result_info, f"错误:\n{error_msg}", is_error=True)
            submit_button.disabled = False
            page.update()
            return

        # 3. 处理 Universe
        univ = []
        if chk_manual_univ.value: # 手动输入模式
            manual_str = txt_manual_univ.value.strip()
            if not manual_str:
                log_message("错误：勾选了手动输入 Universe，但输入框为空。")
                show_info_message(sample_result_info, "错误：手动 Universe 输入不能为空。", is_error=True)
                txt_manual_univ.error_text = "输入不能为空"
                error_occurred = True
            else:
                try:
                    univ_nums = [int(x) for x in manual_str.split()]
                    if len(univ_nums) != n_processed:
                        raise ValueError(f"需要输入 {n_processed} 个数字，实际输入了 {len(univ_nums)} 个。")
                    if len(set(univ_nums)) != len(univ_nums):
                        raise ValueError("输入的数字包含重复项。")
                    if not all(1 <= x <= m_processed for x in univ_nums):
                        raise ValueError(f"输入的数字必须在 1 到 {m_processed} 之间。")
                    univ = sorted(univ_nums)
                    log_message(f"使用手动输入的 Universe: {univ}")
                except ValueError as ve:
                    log_message(f"手动 Universe 输入错误: {ve}")
                    show_info_message(sample_result_info, f"手动 Universe 输入错误: {ve}", is_error=True)
                    txt_manual_univ.error_text = str(ve)
                    error_occurred = True
        else: # 随机生成模式
            univ = sorted(random.sample(range(1, m_processed + 1), n_processed))
            log_message(f"随机生成的 Universe: {univ}")

        if error_occurred:
            submit_button.disabled = False
            page.update()
            return

        # 4. 处理覆盖条件 y
        condition_processed = None # 实际用于计算的 y 值
        max_y_single_j = -1 # 先计算 C(j,s)
        try:
            if j_processed >= s_processed:
                max_y_single_j = comb(j_processed, s_processed)
            else:
                 max_y_single_j = 0
        except ValueError:
             log_message("计算 C(j,s) 出错。")
             show_info_message(sample_result_info, "错误: 计算 C(j,s) 时参数无效。", is_error=True)
             error_occurred = True

        if not error_occurred:
            if chk_specify_y.value: # 手动指定 y
                y_str = txt_specify_y.value.strip()
                if not y_str:
                    log_message("错误：勾选了手动指定 y，但输入框为空。")
                    show_info_message(sample_result_info, "错误：手动 y 值输入不能为空。", is_error=True)
                    txt_specify_y.error_text = "输入不能为空"
                    error_occurred = True
                else:
                    try:
                        specified_y = int(y_str)
                        if not (1 <= specified_y <= max_y_single_j):
                             raise ValueError(f"y 值 ({specified_y}) 必须是 1 到 C({j_processed},{s_processed})={max_y_single_j} 之间的整数。")
                        condition_processed = specified_y
                        log_message(f"使用手动指定的覆盖条件 y = {condition_processed}")
                    except ValueError as ve:
                        log_message(f"手动 y 值输入错误: {ve}")
                        show_info_message(sample_result_info, f"手动 y 值输入错误: {ve}", is_error=True)
                        txt_specify_y.error_text = str(ve)
                        error_occurred = True
            else: # 自动计算 y (按照新的定义，y=C(j,s))
                 condition_processed = max_y_single_j
                 log_message(f"使用自动计算的覆盖条件 y = C(j,s) = {condition_processed}")
                 if condition_processed <= 0:
                      log_message("警告：自动计算的 y <= 0，可能导致无意义的计算。")
                      # 可以选择在这里阻止运行，或者让后端处理

        if error_occurred:
             submit_button.disabled = False
             page.update()
             return

        # --- 到这里所有参数都已准备好 ---
        log_message(f"最终参数: M={m_processed}, N={n_processed}, K={k_processed}, J={j_processed}, S={s_processed}, 使用的y={condition_processed}, Timeout={timeout_seconds}s")
        show_info_message(sample_result_info, f"参数验证通过。\nUniverse: {univ}\ny={condition_processed}\n正在启动计算 (超时: {timeout_seconds} 秒)...")

        # --- 获取下一个运行索引 (x) ---
        params_tuple = (m_processed, n_processed, k_processed, j_processed, s_processed)
        current_run_idx = g_counter.get_next_run_index(*params_tuple)
        log_message(f"参数组合 {params_tuple} 的第 {current_run_idx} 次运行。")

        # --- 创建 Sample 实例并运行计算 ---
        try:
            # 注意：传递给 Sample 的 y 值是实际要使用的覆盖条件
            sample = Sample(m=m_processed, n=n_processed, k=k_processed, j=j_processed, s=s_processed,
                            y=condition_processed, # 传递实际使用的 y
                            run_idx=current_run_idx,
                            timeout=timeout_seconds,
                            rand_instance=random)
            # 手动将确定的 Universe 设置给 sample 实例（因为 Sample 初始化时是随机生成的）
            sample.univ = univ
            log_message("Sample 实例创建成功，开始运行计算...")

            # 启动计算
            sample.run()

            # --- 处理计算结果 ---
            log_message(f"计算完成。结果标识符: {sample.ans}")
            # ... (结果处理和数据库保存逻辑与上个版本相同) ...
            if sample.result and 'alg' in sample.result:
                algorithm_used = sample.result.get('alg', 'N/A')
                num_sets_found = len(sample.sets) if isinstance(sample.sets, list) else 0
                time_taken = sample.result.get('time', 0)
                status = sample.result.get('status', 'Unknown')

                log_message(f"求解算法: {algorithm_used}, 状态: {status}, 耗时: {time_taken:.2f}s, 找到集合数: {num_sets_found}")

                result_text = (
                    f"结果标识 (m-n-k-j-s-x-y'): {str(sample.ans)}\n"
                    f"使用Universe: {sample.univ}\n"
                    f"求解算法: {algorithm_used}\n"
                    f"状态: {status}\n"
                    f"使用的覆盖条件 y: {condition_processed}\n" # 显示实际使用的y
                    f"计算耗时: {time_taken:.2f} 秒\n"
                    f"找到的集合 ({num_sets_found} 个):\n"
                )
                MAX_SETS_TO_DISPLAY = 50
                if num_sets_found > 0 :
                    sets_to_display = sample.sets[:MAX_SETS_TO_DISPLAY]
                    result_text += "\n".join([str(sorted(s)) for s in sets_to_display])
                    if num_sets_found > MAX_SETS_TO_DISPLAY:
                        result_text += f"\n... (还有 {num_sets_found - MAX_SETS_TO_DISPLAY} 个集合未显示)"
                else:
                    result_text += "(无)"

                show_info_message(sample_result_info, result_text)

                # --- 数据库存储 (仅当 K=6 时) ---
                if k_processed == 6:
                    log_message("检测到 K=6，尝试将结果保存到数据库...")
                    try:
                        universe_str = json.dumps(sample.univ)
                        found_sets_str = json.dumps(sample.sets)

                        result_data = {
                            'm': m_processed,
                            'n': n_processed,
                            'k': k_processed,
                            'j': j_processed,
                            's': s_processed,
                            'run_index': current_run_idx,
                            'num_results': num_sets_found,
                            'y_condition': condition_processed, # 保存实际使用的 y 值
                            'algorithm': algorithm_used,
                            'time_taken': time_taken,
                            'universe': universe_str,
                            'sets_found': found_sets_str
                        }
                        save_success = db.save_result(result_data)
                        if save_success:
                            log_message(f"结果已成功保存到数据库 (标识: {sample.ans})。")
                        else:
                            log_message(f"保存结果到数据库时遇到问题 (标识: {sample.ans})，请查看上方日志。")
                    except Exception as db_err:
                        error_msg = f"错误：保存结果到数据库时发生意外失败: {db_err}"
                        log_message(error_msg)
                        print(error_msg)
                        import traceback
                        traceback.print_exc()
                else:
                    log_message(f"K={k_processed} (不是6)，结果未保存到数据库。")
            else:
                 error_msg = f"计算执行完毕，但无法获取有效的算法结果详情。Result: {sample.result}, Ans: {sample.ans}"
                 log_message(error_msg)
                 show_info_message(sample_result_info, error_msg, is_error=True)

        except ValueError as e:
            error_msg = f"创建计算实例时参数错误: {e}"
            log_message(error_msg)
            show_info_message(sample_result_info, error_msg, is_error=True)
        except Exception as e:
            error_msg = f"计算执行过程中发生错误: {e}"
            log_message(error_msg)
            show_info_message(sample_result_info, f"运行时错误: {e}", is_error=True)
            print(f"--- 未捕获的运行时错误 ---")
            import traceback
            traceback.print_exc()
            print(f"--- 错误结束 ---")
        finally:
            # 无论成功失败，重新启用提交按钮
            submit_button.disabled = False
            page.update()

    # 将 on_submit 函数绑定到按钮
    submit_button.on_click = on_submit

    # --- 布局 ---
    page.add(
        ft.Column(
            [
                ft.Text("参数输入", size=16, weight=ft.FontWeight.BOLD),
                ft.Row( # 第一行参数输入
                    [txt_m, txt_n, txt_k],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND # 让控件间距更均匀
                ),
                ft.Row( # 第二行参数输入
                    [txt_j, txt_s, txt_timeout],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND
                ),
                 ft.Divider(),
                 # Universe 输入选项
                 ft.Row([chk_manual_univ, txt_manual_univ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                 # y 值输入选项
                 ft.Row([chk_specify_y, txt_specify_y], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                 ft.Divider(),
                 # 显示 y 相关信息
                 theoretical_y_info,
                 max_single_j_coverage,
                 ft.Divider(),
                 # 操作按钮
                 ft.Row([submit_button, clear_log_button]),
                 # 日志输出
                 ft.Text("运行日志", size=14, weight=ft.FontWeight.BOLD),
                 ft.Container(
                    content=log_output,
                    border=ft.border.all(1, ft.Colors.BLACK12),
                    border_radius=ft.border_radius.all(5),
                    padding=5,
                    expand=True # 让日志区域占据剩余空间
                 ),
                 # 结果输出
                 ft.Text("计算结果", size=14, weight=ft.FontWeight.BOLD),
                 ft.Container(
                     content=sample_result_info,
                     border=ft.border.all(1, ft.Colors.BLACK26),
                     border_radius=ft.border_radius.all(5),
                     padding=10,
                     # height=250 # 可以设置一个最小高度
                 )
            ],
            expand=True,
            scroll=ft.ScrollMode.ADAPTIVE # 允许整个页面在内容过多时滚动
        )
    )

    # --- 应用初始化 ---
    log_message("应用程序启动。")
    log_message(f"数据库文件位于: {os.path.abspath(db.DB_FILE)}")
    # 初始化时计算一次 y 相关信息
    update_y_related_info()
    # 初始化数据库
    try:
        db.setup_database()
        log_message("数据库初始化检查完成。")
    except Exception as init_db_err:
        log_message(f"错误：初始化数据库失败: {init_db_err}")
        print(f"错误：初始化数据库失败: {init_db_err}")
        import traceback
        traceback.print_exc()

# --- 应用入口 ---
if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) > 1:
        custom_db_path = sys.argv[1]
        if os.sep in custom_db_path or custom_db_path.endswith(".db"):
            if os.path.isdir(custom_db_path):
                 db.DB_FILE = os.path.join(custom_db_path, "k6_results.db")
            else:
                 db.DB_FILE = custom_db_path
            print(f"使用数据库路径: {os.path.abspath(db.DB_FILE)}")
        else:
             print(f"警告：命令行参数 '{custom_db_path}' 无效, 使用默认数据库 '{db.DB_FILE}'。")

    # 启动 Flet 应用
    ft.app(target=main)