import os


def extract_gb_data(line):
    gb_index = line.find("after one epoch") + len("after one epoch: ")
    gb_str = line[gb_index:].strip()
    return gb_str


def seconds_to_hours(seconds):
    hours = seconds / 3600
    return round(hours, 2)


def extract_total_time(filename):
    total_time_sum = 0.0
    max_test_acc = 0.0
    min_test_loss = None
    gb_data = 0.00
    epoch = 0.
    with open(filename, 'r') as file:

        lines = file.readlines()
        for line in lines:
            if "Namespace" in line:
                continue

            if "total_time" in line:
                time_index = line.find("total_time") + len("total_time=")
                time_str = line[time_index:].strip()
                time_str = time_str.split(',')[0]
                total_time_sum += float(time_str)
                epoch += 1

            if "max_test_acc" in line:
                max_test_acc_index = line.find("max_test_acc") + len("max_test_acc=")
                max_test_acc_str = line[max_test_acc_index:].strip()
                max_test_acc_str = max_test_acc_str.split(',')[0]
                max_test_acc = float(max_test_acc_str)

            if "test_loss" in line:
                min_test_loss_index = line.find("test_loss") + len("test_loss=")
                min_test_loss_str = line[min_test_loss_index:].strip()
                min_test_loss_str = min_test_loss_str.split(',')[0]
                min_test_loss = float(min_test_loss_str)

            if "after one epoch" in line:
                gb_data = extract_gb_data(line)
                gb_data = float(gb_data[:-2])

    extract_info_file = "info.txt"
    total_time = f"Total Time Sum: {seconds_to_hours(total_time_sum)} h"
    time_pre_epoch = f"Time / epoch: {round(total_time_sum / epoch, 2)} s / epoch" if epoch != 0 else f"Time / epoch: None"
    max_test_acc = f"Max Test Acc: {round(max_test_acc * 100, 2)} %"
    min_test_loss = f"Min Test Loss: {round(min_test_loss, 4)}" if min_test_loss != None else f"Min Test Loss: None"
    memory_use = f"Memory use / Epoch: {round(gb_data, 3)} GB"
    print(total_time)
    print(time_pre_epoch)
    print(max_test_acc)
    print(min_test_loss)
    print(memory_use)

    new_file = os.path.join(os.path.dirname(filename), extract_info_file)
    with open(new_file, "w") as file:
        file.write(total_time + "\n")
        file.write(time_pre_epoch + "\n")
        file.write(max_test_acc + "\n")
        file.write(min_test_loss + "\n")
        file.write(memory_use + "\n")


if __name__ == '__main__':

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_path)
    print("Parent Path:", parent_path)
    print("Current File's Path:", current_path)

    log_dir = os.path.join(parent_path, "logs")
    print(log_dir)

    # directory_folder = "/BPTT_DVSCIFAR10_spiking_vgg11_bn__T10_tau1.1_e300_bs128_SGD_lr0.05_wd0.0005_SG_triangle_drop0.3_losslamb0.05_CosALR_300_amp"

    # file = log_dir + directory_folder + "/args.txt"
    # print(directory_folder)
    # extract_total_time(file)

    for folder_name in os.listdir(log_dir):
        folder_path = os.path.join(log_dir, folder_name)
        file_path = folder_path + "/args.txt"
        print(file_path)
        if os.path.exists(file_path):
            extract_total_time(file_path)
            print("\n")
