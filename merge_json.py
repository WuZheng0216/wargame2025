import json
from tkinter import Tk, filedialog, messagebox

def merge_selected_jsons(output_file="merged_simulation.json"):
    """弹出文件选择框，选择多个 JSON 文件后自动合并"""

    # 关闭主窗口，只保留对话框
    root = Tk()
    root.withdraw()

    # 弹出文件选择框
    file_paths = filedialog.askopenfilenames(
        title="请选择要合并的仿真 JSON 文件",
        filetypes=[("JSON 文件", "*.json")]
    )

    if not file_paths:
        messagebox.showinfo("提示", "未选择任何文件。")
        return

    merged_data = []
    factions = set()

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "sim_time" in data:
                merged_data.append(data)
                if "faction" in data:
                    factions.add(data["faction"])
            else:
                print(f"⚠️ {file_path} 格式不符，已跳过。")
        except Exception as e:
            print(f"❌ 无法读取 {file_path} ：{e}")

    if not merged_data:
        messagebox.showerror("错误", "未找到有效 JSON 文件。")
        return

    # 按 sim_time 排序
    merged_data.sort(key=lambda x: x.get("sim_time", 0))

    summary = {
        "total_files": len(merged_data),
        "factions": list(factions),
        "time_range": [merged_data[0]["sim_time"], merged_data[-1]["sim_time"]],
    }

    merged_output = {
        "session_summary": summary,
        "records": merged_data
    }

    # 选择输出位置
    save_path = filedialog.asksaveasfilename(
        title="保存合并后的文件为",
        defaultextension=".json",
        initialfile=output_file,
        filetypes=[("JSON 文件", "*.json")]
    )

    if not save_path:
        messagebox.showinfo("提示", "未选择保存位置。")
        return

    with open(save_path, "w", encoding="utf-8") as f_out:
        json.dump(merged_output, f_out, ensure_ascii=False, indent=2)

    messagebox.showinfo(
        "合并完成",
        f"成功合并 {len(merged_data)} 个文件！\n"
        f"阵营: {', '.join(summary['factions'])}\n"
        f"时间范围: {summary['time_range'][0]}–{summary['time_range'][1]}\n"
        f"保存路径: {save_path}"
    )


if __name__ == "__main__":
    merge_selected_jsons()
