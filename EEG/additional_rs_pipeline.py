from preproc import clean_raw, ten_sec_epochs
import sys

if __name__ == "__main__":
    # Check if at least one command-line argument is provided
    if len(sys.argv) < 3:
        print("Usage: python additional_rs_pipeline.py <condition> <subject_id>")
    else:
        # Get the 1st and 2nd command-line argument
        arg1 = sys.argv[1]
        subject_id = sys.argv[2]

        if arg1 == 'open':
            condition = 'RESTINGSTATEOPEN'
        elif arg1 == 'close':
            condition = 'RESTINGSTATECLOSE'
        else:
            print('Usage: python additional_rs_pipeline.py <condition> <subject_id>')
            print('Condition must be either "open" or "close"')
            sys.exit()

        clean_raw(subject_id, condition)
        ten_sec_epochs(subject_id, condition)

