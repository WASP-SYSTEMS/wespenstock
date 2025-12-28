"""
Crawl single OSS-Fuzz issue page.
"""

import sys

import requests
from bs4 import BeautifulSoup
from bs4 import Tag
from playwright.sync_api import sync_playwright
from pydantic import BaseModel


class Links(BaseModel):
    """
    Links to testcase and revision.
    """

    testcase: str
    revision: str


class Entry(BaseModel):
    """
    Entry in database.
    """

    name: str
    oss_fuzz_issue: str
    harness_name: str
    vuln_function: str
    vuln_commit: str
    introspector_url: str


class CrawlError(Exception):
    """
    Crawling exception.
    """

    def __init__(self, msg: str):
        super().__init__(msg)


def load_page(url: str) -> BeautifulSoup:
    """
    Load url and execute javascript using playwright. Finally parse to bs4.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle")
        content = page.content()
        browser.close()
    return BeautifulSoup(content, "html.parser")


def get_sinlge(page: BeautifulSoup | Tag, tag: str) -> Tag:
    """Assumes there is a single <tag> and returns it."""
    entries = page.find_all(tag)
    if len(entries) != 1:
        raise CrawlError(f"Try finding single {tag}-tag is ambitious: {len(entries)} found.")
    return entries[0]


def get(tag: Tag, name: str) -> str | None:
    """
    Wrapper of Tag.get for type safety.
    """
    if not (value := tag.get(name)):
        return None
    if not isinstance(value, str):
        return None
    return value


def crawl_links(page: BeautifulSoup) -> Links:
    """Given an OSS-Fuzz issue page, get testcase and revision links."""
    testcase = None
    revision = None
    issue_description = get_sinlge(page, "b-issue-description")
    for link in issue_description.find_all("a"):
        if (href := get(link, "data-b-original-href")) is not None:
            if href.startswith("https://oss-fuzz.com/revisions"):
                if revision is not None:
                    raise CrawlError("More than one Revision link")
                revision = href
            elif href.startswith("https://oss-fuzz.com/download"):
                if testcase is not None:
                    raise CrawlError("More than one Testcase link")
                testcase = href
    if testcase is None or revision is None:
        raise CrawlError("Could not find all links")
    return Links(testcase=testcase, revision=revision)


def crawl_project_name(page: BeautifulSoup) -> str:
    """Given an OSS-Fuzz issue page, get project name."""
    issue_title = get_sinlge(page, "issue-title")
    title = get_sinlge(issue_title, "h3").get_text(strip=True)
    return title[: title.find(":")]


def crawl_harness_name(page: BeautifulSoup) -> str:
    """Given an OSS-Fuzz issue page, get harness name."""
    target = None
    issue_description = get_sinlge(page, "b-issue-description")
    for cell in issue_description.prettify().split("\n"):
        if "Fuzz Target:" in cell:
            if target is not None:
                raise CrawlError("More than one fuzz target.")
            target = cell[cell.find("Fuzz Target:") + len("Fuzz Target:") :].strip()
    if target is None:
        raise CrawlError("Could not find fuzz target.")
    return target


def request_introspector_report(project: str) -> str:
    """Given a project name, find introspector report URL."""
    response = requests.get(  # pylint: disable=missing-timeout
        f"https://introspector.oss-fuzz.com/project-profile?project={project}"
    )
    page = BeautifulSoup(response.text, "html.parser")
    links = []
    for link in page.find_all("a"):
        if (
            (href := get(link, "href")) is not None
            and href.endswith("fuzz_report.html")
            and href.startswith("https://storage.googleapis.com/oss-fuzz-introspector/")
        ):
            links.append(href)
    links = list(set(links))
    if len(links) != 1:
        raise CrawlError(f"Found several introspector report links: {links}")
    return links[0]


def fetch_revision_html(revision: str) -> BeautifulSoup:
    """
    Load revision page.
    This needs some extra care to load the specific table correctly.
    """
    html = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(revision, wait_until="networkidle")
        page.wait_for_selector("table")
        table = page.locator("table").all()
        if len(table) > 1:
            raise CrawlError("More than one table in revision page")
        html = table[0].inner_html()
        browser.close()
    return BeautifulSoup(html, "html.parser")


def crawl_revision_commit(table: BeautifulSoup, project: str) -> str:
    """
    Given the revision table and the project name, find the vulnerable commit.
    """
    matching_rows = []
    for row in table.find_all("tr", class_="body"):
        tds = row.find_all("td")
        if len(tds) != 2:
            continue
        span = tds[0].find("span")
        if span and span.get_text(strip=True).lower() == project:
            matching_rows.append(row)
    if len(matching_rows) > 1:
        raise CrawlError("More than one revision table row with project name")
    row = matching_rows[0]
    link = get_sinlge(row, "a")
    return link.get_text(strip=True)


def crawl(link: str) -> Entry:
    """
    Process a single OSS-Fuzz issue link.
    """
    page = load_page(link)
    links = crawl_links(page)
    print(f"Links: {links}")
    project = crawl_project_name(page)
    print(f"Project: {project}")
    harness = crawl_harness_name(page)
    print(f"Harness: {harness}")
    commit = crawl_revision_commit(fetch_revision_html(links.revision), project)
    print(f"Commit: {commit}")
    introsepctor_report = request_introspector_report(project)
    print(f"Introspector report: {introsepctor_report}")

    return Entry(
        name=project,
        oss_fuzz_issue=link,
        harness_name=harness,
        vuln_function="TODO",
        vuln_commit=commit,
        introspector_url=introsepctor_report,
    )


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print("usage:")
        print(f"python {sys.argv[0]} <link to oss-fuzz issue>")
        sys.exit(1)

    link = sys.argv[1]
    print("Fetching ...")

    entry = crawl(link)

    print(entry)


if __name__ == "__main__":
    main()
