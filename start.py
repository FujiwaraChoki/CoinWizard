import os
import multiprocessing
from termcolor import colored


def start_frontend():
    print(colored('[x] Starting Frontend...', 'blue'))
    os.chdir('Frontend')
    os.system('python3 -m http.server 3000')
    print(colored('[x] Frontend Started...\n', 'blue'))
    print(colored('[x] Open http://localhost:3000', 'green'))


def start_backend():
    print(colored('[x] Starting Backend...', 'blue'))
    os.chdir('Backend')
    print(colored('[x] Installing Requirements...', 'blue'))
    os.system('pip3 install -r requirements.txt')
    print(colored('[x] Requirements Installed...', 'blue'))
    os.system('python3 app.py')
    print(colored('[x] Backend Started...', 'green'))


def main():
    frontend_p = multiprocessing.Process(target=start_frontend)
    backend_p = multiprocessing.Process(target=start_backend)

    # Start the processes
    frontend_p.start()
    backend_p.start()


if __name__ == '__main__':
    main()
