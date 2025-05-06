import time
from locust import HttpUser, task, between
from random import randint, choice

class HotelUser(HttpUser):
    wait_time = between(2,5)

    @task
    def search_hotel(self):
        in_date = randint(9, 23)
        out_date = randint(in_date + 1, 30) 
        lat, lon = HotelUser.get_lat_lon()
        
        self.client.get(f"/hotels?inDate=2015-04-{in_date:02}&outDate=2015-04-{out_date:02}&lat={lat}&lon={lon}", 
                        name="/hotels" ,headers={"Content-Type":"application/x-www-form-urlencoded"})

    @task
    def recommend(self):
        require = choice(["dis", "rate", "price"])
        lat, lon = HotelUser.get_lat_lon()
        self.client.get(f"/recommendations?require={require}&lat={lat}&lon={lon}", 
                        name="/recommendations")# ,headers={"Content-Type":"application/x-www-form-urlencoded"})

    @task
    def reserve(self):
        lat, lon = HotelUser.get_lat_lon()
        in_date = randint(9, 23)
        out_date = in_date + randint(1, 5)
        hotel_id = str(randint(1, 80))
        username, password = HotelUser.get_user()
        num_room = "1"


        self.client.post(f"/reservation?inDate=2015-04-{in_date:02}&outDate=2015-04-{out_date:02}&lat={lat}&lon={lon}&hotelId={hotel_id}&customerName={username}&username={username}&password={password}&number={num_room}",
                        name="/reservation")

    @task
    def user_login(self):
        username, password = HotelUser.get_user()
        self.client.post(f"/user?username={username}&password={password}",
                        name="/user")

    @staticmethod
    def get_lat_lon():
        return 38.0235 + (randint(0, 481) - 240.5)/1000.0, -122.095 + (randint(0, 325) - 157.0)/1000.0

    @staticmethod
    def get_user():
        id = randint(0,500)
        return f"Cornell_{id}", ''.join(str(i) for i in range(0,9))
