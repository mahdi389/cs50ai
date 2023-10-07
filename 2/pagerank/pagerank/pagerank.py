import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    #making a dic with key values
    d = {}
    d.update(corpus)

    #if we choose on rnadom between all pages
    for i in range (1 , len(corpus)+1):
        d[f'{i}.html'] = (1-damping_factor)/len(corpus)
    
    
    #choosing from that page
    for i in corpus[page]:
        d[i] += damping_factor/len(corpus[page])
    
    return(d)


def sample_pagerank(corpus, damping_factor, n):
    
    ourw = transition_model(corpus , random.choice(list(corpus.keys())) , damping_factor)
    pages = list(ourw.keys())
    pagesw = list(ourw.values())
    sample = random.choices(pages, weights=pagesw, k=n)

    ans = {}
    ans.update(corpus)
    
    for i in range (1 , len(corpus)+1):
        ans[f'{i}.html'] = sample.count(f'{i}.html')/n

    return(ans)





def iterate_pagerank(corpus, damping_factor):
    ans = {}
    ans.update(corpus)
    pages = list(corpus.keys())
    for i in range(0,len(pages)):
        ans[pages[i]] = 1/float(len(pages))
        ans[pages[i]] = round(ans[pages[i]] , 3)



    close = False
    change = []
    for i in range ( 0 , len(pages)):
        change.append(0)
    while close == False:

        for i in range(0 , len(pages)):
            past = ans[pages[i]]
            ans[pages[i]] = (1-damping_factor) / float(len(pages))
            add = 0
            for j in range ( 0 , len(pages)):
                if pages[i] in corpus[pages[j]]:
                     add += ans[pages[j]] / float(len(corpus[pages[j]]))
            add = add*damping_factor
            ans[pages[i]] += add
            ans[pages[i]] = round(ans[pages[i]] , 3)
            change[i] = abs(ans[pages[i]] - past)

        close = True
        for i in range ( 0 , len(change)):
            if change[i] > 0.001:
                close = False

    return (ans)


if __name__ == "__main__":
    main()
