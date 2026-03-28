import re
import os

# 配置：输入/输出文件名
INPUT_LOG = "log_50_4090_full copy.txt"   # 左边的日志文件
OUTPUT_LOG = "log_50_4090_converted.txt"  # 输出的新日志

def convert_line(line):
    # 匹配格式：100/80000 (0.1%) → 拆成 100/80000 | 0.1%
    pattern = r"(\d+/\d+) \((\d+\.\d+%)\)"
    match = re.search(pattern, line)
    if match:
        step_part = match.group(1)
        percent_part = match.group(2)
        # 替换成两列格式
        new_line = re.sub(pattern, f"{step_part} | {percent_part}", line)
        return new_line
    else:
        return line

def convert_log():
    if not os.path.exists(INPUT_LOG):
        print(f"❌ 找不到输入文件：{INPUT_LOG}")
        return

    with open(INPUT_LOG, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    new_lines = []
    for line in lines:
        new_line = convert_line(line)
        new_lines.append(new_line)

    with open(OUTPUT_LOG, "w", encoding="utf-8") as f_out:
        f_out.writelines(new_lines)

    print(f"✅ 转换完成！新文件：{OUTPUT_LOG}")
    print(f"📝 格式变化：'100/80000 (0.1%)' → '100/80000 | 0.1%'")

if __name__ == "__main__":
    convert_log()