import os
import glob
import math
import time

import matplotlib.pyplot as plt 
import numpy as np

from scipy.stats import weibull_min

from locust import HttpUser, task, between, LoadTestShape
from random import randint, choice
from locust import events
from locust.runners import MasterRunner

from weibull import get_n_weibull_variables, compute_weibull_scale

PATH = "./locust/"
log_file=""

@events.test_stop.add_listener
def on_test_stop(environment, **_kwargs):
    if not isinstance(environment.runner, MasterRunner):
        return

    label = environment.parsed_options.csv_prefix
    total_file = f"/home/pager/Documents/closing-the-loop/{label}_responce_log.csv"
    csv_log_files = glob.glob("/home/pager/Documents/closing-the-loop/.response_times_*.csv")
    
    with open(total_file, "w") as f:
        f.write("request_type,name,response_time,response_length,status_code\n")
        for log_name in csv_log_files:
            with open(log_name, 'r') as in_f:
                f.write(in_f.read())

@events.quitting.add_listener
def on_quit(environment):
    if isinstance(environment.runner, MasterRunner):
        return
    
    global log_file
    os.remove(log_file)


@events.test_start.add_listener
def on_test_start(environment, **_kwargs):
    global log_file
    worker_id = os.getpid()  # or use uuid.uuid4() for uniqueness
    log_file = f"/home/pager/Documents/closing-the-loop/.response_times_{worker_id}.csv"

@events.request.add_listener
def log_request(request_type, name, response_time, response_length, response, **kwargs):
    global log_file


    with open(log_file, "a") as f:
        f.write(f"{request_type},{name},{response_time},{response_length}, {response.status_code}\n")


class HotelUser(HttpUser):
    wait_time = between(3,5)

    @task(60)
    def search_hotel(self):
        in_date = randint(9, 23)
        out_date = randint(in_date + 1, 30) 
        lat, lon = HotelUser.get_lat_lon()
        
        self.client.get(f"/hotels?inDate=2015-04-{in_date:02}&outDate=2015-04-{out_date:02}&lat={lat}&lon={lon}", 
                        name="/hotels") # ,headers={"Content-Type":"application/x-www-form-urlencoded"})

    @task(38)
    def recommend(self):
        require = choice(["dis", "rate", "price"])
        lat, lon = HotelUser.get_lat_lon()
        self.client.get(f"/recommendations?require={require}&lat={lat}&lon={lon}", 
                        name="/recommendations") # ,headers={"Content-Type":"application/x-www-form-urlencoded"})

    @task(1)
    def reserve(self):
        lat, lon = HotelUser.get_lat_lon()
        in_date = randint(9, 23)
        out_date = in_date + randint(1, 5)
        hotel_id = str(randint(1, 80))
        username, password = HotelUser.get_user()
        num_room = "1"


        self.client.post(f"/reservation?inDate=2015-04-{in_date:02}&outDate=2015-04-{out_date:02}&lat={lat}&lon={lon}&hotelId={hotel_id}&customerName={username}&username={username}&password={password}&number={num_room}",
                        name="/reservation")

    @task(1)
    def user_login(self):
        username, password = HotelUser.get_user()
        self.client.post(f"/user?username={username}&password={password}",
                        name="/user")

    @staticmethod
    def get_lat_lon():
        return 38.0235 + float(randint(0, 481) - 240.5)/1000.0, -122.095 + float(randint(0, 325) - 157.0)/1000.0

    @staticmethod
    def get_user():
        id = randint(0,500)
        return f"Cornell_{id}", ''.join(str(i) for i in range(0,9))





@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--w-shape", type=int,is_required=False, default=1)
    parser.add_argument("--w-mean", type=int, is_required=False, default=150)
    parser.add_argument("--w-user-min", type=int, is_required=False, default=100)
    parser.add_argument("--w-user-max", type=int, is_required=False, default=1000)
    parser.add_argument("--w-dt", type=int, is_required=False, default=20)

class WeibullShape(LoadTestShape):
    stages = []
    use_common_options = True

    def plot(self, tmin, tmax, shape_k, scale_lambda, T, N, L):
        # Plot histogram of samples
        plt.figure(figsize=(10, 6))
        bins = np.linspace(tmin, tmax, 100)
        plt.hist(L, bins=bins, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label="Truncated Weibull Samples")

        # Plot analytical truncated PDF
        x_vals = np.linspace(tmin, tmax, 500)
        pdf_vals = weibull_min.pdf(x_vals, c=shape_k, scale=scale_lambda)
        cdf_min = weibull_min.cdf(tmin, c=shape_k, scale=scale_lambda)
        cdf_max = weibull_min.cdf(tmax, c=shape_k, scale=scale_lambda)
        truncated_pdf = pdf_vals / (cdf_max - cdf_min)

        plt.plot(x_vals, truncated_pdf, 'r--', lw=2, label="Truncated Weibull PDF")
        plt.title("Truncated Weibull Distribution Samples")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(PATH + f"user_count_distribution_{time.time()}.png")
        plt.clf()

        # plot users per second
        plt.plot(np.linspace(0, T, N), L)
        plt.title("Truncated Weibull Load generation in Users per Second.")
        plt.xlabel("time [S]")
        plt.ylabel("users [#]")
        plt.grid(True)

        plt.savefig(PATH + f"users_over_time_{time.time()}.png")
        plt.clf()
   

    def tick(self):
        # build stages on the first tick.
        if self.stages == []:
            w_shape  = self.runner.environment.parsed_options.w_shape
            w_mean  = self.runner.environment.parsed_options.w_mean
            U_min  = self.runner.environment.parsed_options.w_user_min
            U_max  = self.runner.environment.parsed_options.w_user_max
            T  = self.runner.environment.parsed_options.run_time
            dt  = self.runner.environment.parsed_options.w_dt
            

            ## magic-kit
            _lambda = compute_weibull_scale(w_mean, w_shape)
            N = int(T/dt)
            L = get_n_weibull_variables(w_shape, _lambda, U_min, U_max, N)

            l_prev = 0
            for s, l in zip(range(dt,T+dt,dt), L):
                self.stages.append({"start": s, "load": l, "rate": int(math.ceil(abs((l_prev - l)/dt)))})
                l_prev = l
            
            self.plot(U_min, U_max, w_shape, _lambda, T, N, L)
         
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["start"]:
                return (stage["load"], stage["rate"])

        return None