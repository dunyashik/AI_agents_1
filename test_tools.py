from tools import web_search, page_find, ddg_search, calculator, python_exec


def test_web_search():
    res = web_search("2026 Winter Olympics")
    assert len(res) > 20
    assert "Olympics" in res

    res_empty = web_search("xzcvxzcvqwerqwerasdfasdf12345678")
    assert res_empty == "nothing found"

    res_err = web_search("")
    assert isinstance(res_err, str)


def test_page_find():
    res = page_find("2026 Winter Olympics", "Milano Cortina")
    assert len(res) > 20
    assert "Milano" in res or "Cortina" in res or "Olympics" in res

    res_empty = page_find("2026 Winter Olympics", "randomunknownkeyword12345")
    assert res_empty == "keywords not found on the page"

    res_err = page_find("NonExistentPage_999999_XYZ", "Milano")
    assert res_err == "no such page"


def test_ddg_search():
    res = ddg_search("Python programming language")
    assert len(res) > 10

    res_empty = ddg_search("asdfqwerzxcv1234567890nonexistentquery")
    assert res_empty == "nothing found" or isinstance(res_empty, str)

    res_err = ddg_search("")
    assert isinstance(res_err, str)


def test_calculator():
    res = calculator("17*23+5")
    assert res == "396"

    res_edge = calculator("0*100")
    assert res_edge == "0"

    res_err = calculator("__import__('os')")
    assert "error" in res_err


def test_python_exec():
    res = python_exec("print(sum(range(10)))")
    assert res == "45"

    res_empty = python_exec("x = 1")
    assert res_empty == "code ran but printed nothing"

    res_err = python_exec("1/0")
    assert "ZeroDivisionError" in res_err

    res_timeout = python_exec("import time; time.sleep(10)")
    assert res_timeout == "code exceeded the 5 second limit"


if __name__ == "__main__":
    test_web_search()
    test_page_find()
    test_ddg_search()
    test_calculator()
    test_python_exec()
    print("All tool tests passed successfully.")