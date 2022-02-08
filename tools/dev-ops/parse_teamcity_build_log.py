import argparse


def _parse(path_to_log: str):
    with open(path_to_log, 'r') as f:
        lines = f.readlines()

    config_sentence = 'Config:'
    done_sentence = 'Done! Elapsed'

    keywords = ['Error',
                config_sentence, 'task_name', 'path_to_data_dir', 'algorithm_name', 'backbone_name',
                done_sentence]

    for i, line in enumerate(lines):
        for keyword in keywords:
            if keyword in line:
                if keyword == config_sentence:
                    print('\n', '=' * 80, '\n')

                if keyword == done_sentence:
                    for j in range(i - 1, 0, -1):
                        if 'Start evaluating for test set' in lines[j]:
                            for k in range(j + 1, i):
                                print(lines[k].strip())
                            break

                print(line.strip())
                break


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('log', type=str, help='path to log')
        args = parser.parse_args()

        path_to_log = args.log
        _parse(path_to_log)

    main()
