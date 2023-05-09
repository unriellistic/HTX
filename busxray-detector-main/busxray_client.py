import argparse, signal, time

from busxray_observer import BusxrayObserver

# handler to exit when CTRL+C is received
def signal_handler(signal, frame):
    print("Exiting.")
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Source folder to watch for changes", default=".")
    parser.add_argument("--output", help="Output folder to save JSON predictions to", default="output")
    parser.add_argument("--url", help="Target URL to send requests to", default="http://localhost:8000")
    args = parser.parse_args()

    # observe the folder for changes
    observer = BusxrayObserver(args.source, args.output, args.url)
    observer.start()

    # wait for CTRL+C
    print("Press CTRL+C to exit.")
    signal.signal(signal.SIGINT, signal_handler)
    while True:
        time.sleep(0.1)