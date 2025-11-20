import bs4


def parse_honor(html_path, keywords=None):
    """
    Find class="count", which contains many <div> elements with class="list",
    each containing <img> <div class="list_count_title> <div class="list_desc">.
    Return another format like:
        <div class="honor-card">
            <div class="icon-container">
                <img src="https://example.com/icon.png" alt="Icon" class="icon">
            </div>
            <div class="honor-details">
                <div class="award-title">XXXXXXXXXXXXXX</div>
                <div class="authors">XXX, XXX, XXX, XXX, XXX</div>
            </div>
        </div>
    """
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    soup = bs4.BeautifulSoup(html, 'html.parser')
    count_div = soup.find('div', class_='count')
    if not count_div:
        return []

    honor_list = []
    for item in count_div.find_all('div', class_='list'):
        img = item.find('img')['src'] if item.find('img') else ''
        title = item.find('div', class_='list_count_title').get_text(strip=True) if item.find('div', class_='list_count_title') else ''
        desc = item.find('div', class_='list_desc').get_text(strip=True) if item.find('div', class_='list_desc') else ''

        if keywords:
            if not any(keyword in desc for keyword in keywords):
                continue
        
        honor_list.append({
            'img': img,
            'title': title,
            'desc': desc
        })
        
        output_html = ''
        for honor in honor_list:
            output_html += f"""
            <div class="honor-card">
                <div class="icon-container">
                    <img src="{honor['img']}" alt="Icon" class="icon">
                </div>
                <div class="honor-details">
                    <div class="award-title">{honor['title']}</div>
                    <div class="authors">{honor['desc']}</div>
                </div>
            </div>
            """
    return output_html.strip()


def parse_paper(html_path, keywords=None):
    """
    Find class="list", which contains many <div> elements with class="desc",
    each containing <div class="brief">.
    Return another format like:
        <div class="honor-card">
            <p>XXXXXXXXXXXXXX</p>
        </div>
    """
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    soup = bs4.BeautifulSoup(html, 'html.parser')

    paper_list = []
    for item in soup.find_all('div', class_='desc'):
        brief = item.find('div', class_='brief').get_text(strip=True) if item.find('div', class_='brief') else ''

        if keywords:
            if not any(keyword in brief for keyword in keywords):
                continue
        
        paper_list.append({
            'brief': brief
        })
        
        output_html = ''
        for paper in paper_list:
            output_html += f"""
            <div class="honor-card">
                {paper['brief']}
            </div>
            """
    return output_html.strip()


def output_html(html_content, output_path):
    """
    Write the HTML content to a file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(html_content)


# print(parse_honor('html/honor.html', ['Jingkuan Song', 'Heng Tao Shen']))
output_html((parse_paper('html/paper.html', ['Jingkuan Song', 'Heng Tao Shen'])), 'html/paper_output.html')
