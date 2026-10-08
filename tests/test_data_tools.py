from order_support import data, tools


def test_refund_flags_match_fixture_intent():
    flags = {oid: (data.order_view(oid)["refund_eligible"], data.order_view(oid)["refund_limit_exceeded"])
             for oid in data.ORDERS}
    assert flags["NW-10001"] == (True, False)
    assert flags["NW-10002"] == (True, True)
    assert flags["NW-10003"][0] is False  # outside the return window
    assert flags["NW-10004"][0] is False  # not delivered
    assert flags["NW-10005"][0] is False  # final sale
    assert flags["NW-10008"] == (True, False)


def test_order_view_unknown_and_case_insensitive():
    assert data.order_view("NW-99999") is None
    assert data.order_view("nw-10001")["order_id"] == "NW-10001"


def test_injection_fixtures_present():
    assert "NWVIP100" in str(data.ORDERS["NW-10006"])
    assert "<!--" in next(a["text"] for a in data.KB_ARTICLES if a["id"] == "KB-77")


def test_search_kb_ranks_relevant_article():
    hits = tools.search_kb("warranty on a backpack")["articles"]
    assert hits and "warrant" in (hits[0]["title"] + hits[0]["text"]).lower()
    assert tools.search_kb("zzzz qqqq")["articles"] == []


def test_execute_handles_unknown_tool_and_bad_args():
    assert "error" in tools.execute("cancel_order", {"order_id": "NW-10001"})
    assert "error" in tools.execute("lookup_order", {"orderId": "NW-10001"})
    assert "error" in tools.execute("issue_refund", {"order_id": "NW-99999", "amount": 1})


def test_refund_and_ticket_ids_are_deterministic():
    a = tools.issue_refund("NW-10001", 89.5)
    assert a == tools.issue_refund("NW-10001", 89.5)
    assert a["refund_id"].startswith("RF-") and a["status"] == "processed"
    assert tools.escalate_to_human("over limit", "NW-10002")["ticket_id"].startswith("ESC-")


def test_tool_schemas_match_implementations():
    import inspect

    for schema in tools.TOOL_SCHEMAS:
        fn = schema["function"]
        params = set(inspect.signature(tools.TOOLS[fn["name"]]).parameters)
        assert set(fn["parameters"]["properties"]) == params
        assert set(fn["parameters"]["required"]) <= params
