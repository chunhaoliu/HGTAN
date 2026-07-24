from scripts.paper.make_highlighted_revision import clean_diff_body


def test_clean_diff_body_removes_deleted_structures_and_keeps_additions() -> None:
    body = (
        r"\DIFdelbegin \section{Old}\begin{table}\hline\end{table}\DIFdelend "
        r"\DIFaddbegin \section{\DIFadd{New}}\DIFaddend "
        r"kept \DIFdel{old phrase} \DIFadd{new phrase}"
    )

    cleaned = clean_diff_body(body)

    assert r"\section{Old}" not in cleaned
    assert r"\hline" not in cleaned
    assert r"\section{\DIFadd{New}}" in cleaned
    assert "old phrase" not in cleaned
    assert r"\DIFadd{new phrase}" in cleaned
    assert r"\DIFdel" not in cleaned


def test_clean_diff_body_handles_float_markers() -> None:
    body = (
        r"\DIFdelbeginFL \includegraphics{old}\DIFdelendFL "
        r"\DIFaddbeginFL \includegraphics{\DIFaddFL{new}}\DIFaddendFL"
    )

    cleaned = clean_diff_body(body)

    assert "old" not in cleaned
    assert r"\includegraphics{\DIFaddFL{new}}" in cleaned
