import unittest
from nitwit.router import Endpoint, route, STAGE_DEFAULTS


class TestRouter(unittest.TestCase):
    def test_chat_routes_to_cpu_4b_when_healthy(self):
        ep = route("chat", health=lambda url: True)
        self.assertEqual(ep.base_url, STAGE_DEFAULTS["chat"].base_url)
        self.assertEqual(ep.model, STAGE_DEFAULTS["chat"].model)
        self.assertEqual(ep.extra_body.get("chat_template_kwargs"), {"enable_thinking": False})

    def test_code_is_the_gpu_coder(self):
        ep = route("code", health=lambda url: True)
        self.assertIn("8080", ep.base_url)

    def test_falls_back_to_coder_when_endpoint_down(self):
        # chat endpoint down -> fall back to the GPU coder, no no-think extra_body
        ep = route("chat", health=lambda url: "8080" in url)  # only the coder is up
        self.assertIn("8080", ep.base_url)
        self.assertEqual(ep.extra_body, {})

    def test_verify_routes_to_4b(self):
        ep = route("verify", health=lambda url: True)
        self.assertEqual(ep.base_url, STAGE_DEFAULTS["verify"].base_url)

    def test_default_health_never_raises(self):
        from nitwit.router import _default_health
        self.assertIsInstance(_default_health("http://127.0.0.1:9"), bool)  # unreachable -> False, no raise


if __name__ == "__main__":
    unittest.main()
