from locust import HttpUser, task, between


class UAAG2User(HttpUser):
    wait_time = between(1, 2)

    @task
    def health_check(self):
        self.client.get("/health")
