from __future__ import annotations

import unittest
from unittest import mock

from botchess.solutions.frequency import train as train_frequency
from botchess.web import browser_url, create_app


class WebInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        examples = (
            [{"move_uci": "e2e4", "player_rating": 1200}] * 30
            + [{"move_uci": "e7e5", "player_rating": 1200}] * 20
            + [{"move_uci": "d8h4", "player_rating": 1200}] * 10
        )
        self.model = train_frequency(examples)
        self.app = create_app(self.model, config={"TESTING": True})
        self.client = self.app.test_client()

    def test_page_and_local_assets_are_served(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Jogar contra o botChess", page.data)
        self.assertIn(b'id="play-mode"', page.data)
        self.assertIn(b'<option value="coherent" selected>', page.data)
        self.assertIn(b'id="bot-decision"', page.data)
        self.assertIn("frame-ancestors 'none'", page.headers["Content-Security-Policy"])
        style = self.client.get("/static/style.css")
        script = self.client.get("/static/app.js")
        favicon = self.client.get("/static/favicon.svg")
        try:
            self.assertEqual(style.status_code, 200)
            self.assertEqual(script.status_code, 200)
            self.assertEqual(favicon.status_code, 200)
        finally:
            style.close()
            script.close()
            favicon.close()

    def test_white_player_starts_and_bot_answers_a_legal_move(self) -> None:
        created = self.client.post("/api/games", json={"human_color": "white"})
        self.assertEqual(created.status_code, 201)
        state = created.get_json()
        self.assertEqual(state["mode"], "imitator")
        self.assertEqual(state["turn"], "white")
        self.assertIn("e2e4", state["legal_moves"])

        played = self.client.post(
            f"/api/games/{state['id']}/moves", json={"uci": "e2e4"}
        )
        self.assertEqual(played.status_code, 200)
        after = played.get_json()
        self.assertEqual([move["uci"] for move in after["history"]], ["e2e4", "e7e5"])
        self.assertEqual(after["bot_move"]["uci"], "e7e5")
        self.assertEqual(after["bot_move"]["style_rank"], 1)
        self.assertEqual(after["bot_move"]["selection_reason"], "imitation")
        self.assertFalse(after["bot_move"]["coherence_applied"])
        self.assertEqual(after["turn"], "white")

    def test_requested_play_mode_is_stored_for_the_whole_game(self) -> None:
        for mode in ("imitator", "coherent"):
            with self.subTest(mode=mode):
                created = self.client.post(
                    "/api/games",
                    json={"human_color": "white", "mode": mode},
                )
                self.assertEqual(created.status_code, 201)
                state = created.get_json()
                self.assertEqual(state["mode"], mode)
                current = self.client.get(f"/api/games/{state['id']}").get_json()
                self.assertEqual(current["mode"], mode)

    def test_black_player_receives_the_bots_opening_move(self) -> None:
        created = self.client.post("/api/games", json={"human_color": "black"})
        self.assertEqual(created.status_code, 201)
        state = created.get_json()
        self.assertEqual(state["history"][0]["actor"], "bot")
        self.assertEqual(state["history"][0]["uci"], "e2e4")
        self.assertEqual(state["turn"], "black")
        self.assertTrue(state["legal_moves"])

    def test_illegal_move_does_not_change_the_position(self) -> None:
        state = self.client.post(
            "/api/games", json={"human_color": "white"}
        ).get_json()
        original_fen = state["fen"]
        rejected = self.client.post(
            f"/api/games/{state['id']}/moves", json={"uci": "e2e5"}
        )
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(rejected.get_json()["error"]["code"], "invalid_move")
        current = self.client.get(f"/api/games/{state['id']}").get_json()
        self.assertEqual(current["fen"], original_fen)
        self.assertEqual(current["history"], [])

    def test_checkmate_finishes_the_game_without_an_extra_move(self) -> None:
        state = self.client.post(
            "/api/games", json={"human_color": "white"}
        ).get_json()
        first = self.client.post(
            f"/api/games/{state['id']}/moves", json={"uci": "f2f3"}
        ).get_json()
        self.assertEqual(first["bot_move"]["uci"], "e7e5")
        finished = self.client.post(
            f"/api/games/{state['id']}/moves", json={"uci": "g2g4"}
        ).get_json()
        self.assertTrue(finished["game_over"])
        self.assertEqual(finished["result"], "0-1")
        self.assertEqual(finished["termination"], "CHECKMATE")
        self.assertEqual(finished["history"][-1]["uci"], "d8h4")
        self.assertIn("xeque-mate", finished["message"])

    def test_coherent_api_reports_when_it_replaces_the_style_top_one(self) -> None:
        tactical_model = train_frequency(
            [{"move_uci": "e7e5", "player_rating": 1200}] * 120
            + [{"move_uci": "a7a6", "player_rating": 1200}] * 100
            + [{"move_uci": "d8h4", "player_rating": 1200}] * 90
        )
        client = create_app(tactical_model, config={"TESTING": True}).test_client()
        state = client.post(
            "/api/games",
            json={"human_color": "white", "mode": "coherent"},
        ).get_json()
        state = client.post(
            f"/api/games/{state['id']}/moves", json={"uci": "f2f3"}
        ).get_json()
        self.assertEqual(state["bot_move"]["uci"], "e7e5")
        finished = client.post(
            f"/api/games/{state['id']}/moves", json={"uci": "g2g4"}
        ).get_json()
        self.assertEqual(finished["bot_move"]["uci"], "d8h4")
        self.assertEqual(finished["bot_move"]["style_rank"], 2)
        self.assertEqual(finished["bot_move"]["selection_reason"], "mate_in_one")
        self.assertTrue(finished["bot_move"]["coherence_applied"])
        self.assertTrue(finished["game_over"])

    def test_claimable_threefold_repetition_does_not_end_automatically(self) -> None:
        repetition_model = train_frequency(
            [{"move_uci": "g8f6", "player_rating": 1200}] * 20
            + [{"move_uci": "f6g8", "player_rating": 1200}] * 10
        )
        client = create_app(
            repetition_model, config={"TESTING": True}
        ).test_client()
        state = client.post(
            "/api/games", json={"human_color": "white"}
        ).get_json()
        for move in ("g1f3", "f3g1", "g1f3", "f3g1"):
            state = client.post(
                f"/api/games/{state['id']}/moves", json={"uci": move}
            ).get_json()
        self.assertFalse(state["game_over"])
        self.assertIsNone(state["result"])
        self.assertEqual(len(state["history"]), 8)

    def test_invalid_requests_return_structured_errors(self) -> None:
        missing_color = self.client.post("/api/games", json={})
        self.assertEqual(missing_color.status_code, 400)
        self.assertEqual(
            missing_color.get_json()["error"]["code"], "invalid_request"
        )
        invalid_json = self.client.post(
            "/api/games", data="não é JSON", content_type="application/json"
        )
        self.assertEqual(invalid_json.status_code, 400)
        missing_game = self.client.get("/api/games/inexistente")
        self.assertEqual(missing_game.status_code, 404)
        invalid_mode = self.client.post(
            "/api/games",
            json={"human_color": "white", "mode": "aleatorio"},
        )
        self.assertEqual(invalid_mode.status_code, 400)
        self.assertEqual(
            invalid_mode.get_json()["error"]["code"], "invalid_request"
        )

    def test_model_failure_returns_a_structured_server_error(self) -> None:
        with (
            mock.patch.object(self.model, "rank", side_effect=RuntimeError("falha")),
            self.assertLogs(self.app.logger, level="ERROR"),
        ):
            response = self.client.post(
                "/api/games", json={"human_color": "black"}
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"]["code"], "bot_failure")

    def test_model_failure_rolls_back_the_human_move(self) -> None:
        state = self.client.post(
            "/api/games", json={"human_color": "white"}
        ).get_json()
        with (
            mock.patch.object(self.model, "rank", side_effect=RuntimeError("falha")),
            self.assertLogs(self.app.logger, level="ERROR"),
        ):
            response = self.client.post(
                f"/api/games/{state['id']}/moves", json={"uci": "e2e4"}
            )
        self.assertEqual(response.status_code, 500)
        current = self.client.get(f"/api/games/{state['id']}").get_json()
        self.assertEqual(current["fen"], state["fen"])
        self.assertEqual(current["history"], [])

    def test_browser_url_supports_loopback_ipv4_and_ipv6(self) -> None:
        self.assertEqual(browser_url("127.0.0.1", 8000), "http://127.0.0.1:8000")
        self.assertEqual(browser_url("::1", 8000), "http://[::1]:8000")
        self.assertEqual(browser_url("0.0.0.0", 8000), "http://127.0.0.1:8000")

    def test_game_can_be_removed(self) -> None:
        state = self.client.post(
            "/api/games", json={"human_color": "white"}
        ).get_json()
        deleted = self.client.delete(f"/api/games/{state['id']}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(
            self.client.get(f"/api/games/{state['id']}").status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
