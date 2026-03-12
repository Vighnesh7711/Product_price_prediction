"""
product_scraper.py
==================
Scrape product details from e-commerce URLs (Flipkart, Amazon, generic).
Uses sessions with retries, realistic headers, and multiple fallback strategies.
"""

import re
import json
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from curl_cffi import requests


class ProductScraper:
    def __init__(self):
        # We don't need a complex session setup anymore, 
        # curl_cffi handles the TLS fingerprint automatically
        pass

    # ── public entry point ────────────────────────────────────────
    def scrape(self, url: str) -> dict:
        """Detect site and scrape product details."""
        domain = urlparse(url).netloc.lower()

        try:
            # Impersonate a real browser TLS fingerprint
            resp = requests.get(
                url, 
                impersonate="chrome110",
                timeout=25, 
                allow_redirects=True
            )
            # Raise an exception if the status code is bad (e.g. 529, 403)
            if resp.status_code >= 400:
                raise Exception(f"HTTP Error {resp.status_code}: {resp.reason}")
            
            soup = BeautifulSoup(resp.text, 'lxml')
        except Exception as e:
            err_str = str(e).lower()
            if 'timeout' in err_str or 'connection' in err_str or 'http error 529' in err_str or 'http error 403' in err_str:
                return self._timeout_result(url, domain)
            
            return {'error': str(e), 'url': url, 'name': '', 'price': 0,
                    'scrape_failed': True,
                    'fail_reason': f'Could not access the page: {e}'}

        if 'flipkart' in domain:
            return self._flipkart(soup, url)
        if 'amazon' in domain:
            return self._amazon(soup, url)
        return self._generic(soup, url)

    # ── Flipkart ──────────────────────────────────────────────────
    def _flipkart(self, soup, url):
        d = {'source': 'flipkart', 'url': url, 'scrape_failed': False}

        # Name — try multiple selectors
        for sel in ['span.VU-ZEz', 'span.B_NuCI', 'h1 span', 'h1']:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                d['name'] = el.get_text(strip=True)
                break
        else:
            d['name'] = self._og_value(soup, 'og:title') or 'Unknown Product'

        # Price
        for sel in ['div.Nx9bqj.CxhGGd', 'div._30jeq3._16Jk6d', 'div._30jeq3']:
            el = soup.select_one(sel)
            if el:
                d['price'] = self._parse_price(el)
                break
        if not d.get('price'):
            d['price'] = self._price_from_meta(soup)

        # Original price
        for sel in ['div.yRaY8j.A6\\+E6v', 'div._3I9_wc._2p6lqe', 'div.yRaY8j']:
            el = soup.select_one(sel)
            if el:
                d['original_price'] = self._parse_price(el)
                break

        # Rating
        for sel in ['div.XQDdHH', 'div._3LWZlK']:
            el = soup.select_one(sel)
            if el:
                try:
                    d['rating'] = float(el.get_text(strip=True).split()[0])
                except (ValueError, IndexError):
                    pass
                break

        # Description
        for sel in ['div._1mXcCf', 'div._1AN87F', 'div.RmoJUa']:
            el = soup.select_one(sel)
            if el:
                d['description'] = el.get_text(strip=True)[:500]
                break
        if not d.get('description'):
            d['description'] = self._og_value(soup, 'og:description') or ''

        # Image
        el = (soup.select_one('img._396cs4._2amPTt._3qGmMb')
              or soup.select_one('img._396cs4')
              or soup.select_one('img.DByuf4'))
        d['image'] = el.get('src', '') if el else (self._og_value(soup, 'og:image') or '')

        # Category breadcrumbs
        crumbs = soup.select('a._2whKao') or soup.select('div._1MR4o5 a')
        if crumbs:
            d['category_path'] = ' >> '.join(c.get_text(strip=True) for c in crumbs)

        # Specifications
        d['specifications'] = self._extract_specs(soup)
        return d

    # ── Amazon ────────────────────────────────────────────────────
    def _amazon(self, soup, url):
        d = {'source': 'amazon', 'url': url, 'scrape_failed': False}

        el = soup.select_one('#productTitle')
        d['name'] = el.get_text(strip=True) if el else (
            self._og_value(soup, 'og:title') or 'Unknown Product')

        # Price — try several selectors
        for sel in ['span.a-price-whole', '#priceblock_ourprice',
                    '#priceblock_dealprice', '.a-price .a-offscreen']:
            el = soup.select_one(sel)
            if el:
                d['price'] = self._parse_price(el)
                break
        if not d.get('price'):
            d['price'] = self._price_from_meta(soup)

        # Rating
        el = soup.select_one('#acrPopover') or soup.select_one('span.a-icon-alt')
        if el:
            m = re.search(r'(\d+\.?\d*)', el.get_text())
            if m:
                d['rating'] = float(m.group(1))

        # Reviews
        el = soup.select_one('#acrCustomerReviewText')
        if el:
            m = re.search(r'(\d[\d,]*)', el.get_text())
            if m:
                d['reviews'] = int(m.group(1).replace(',', ''))

        # Description
        el = soup.select_one('#productDescription') or soup.select_one('#feature-bullets')
        d['description'] = el.get_text(strip=True)[:500] if el else (
            self._og_value(soup, 'og:description') or '')

        # Image
        el = soup.select_one('#landingImage') or soup.select_one('#imgBlkFront')
        d['image'] = el.get('src', '') if el else (self._og_value(soup, 'og:image') or '')

        # Category
        crumbs = soup.select('#wayfinding-breadcrumbs_container li a')
        if crumbs:
            d['category_path'] = ' >> '.join(c.get_text(strip=True) for c in crumbs)

        # Brand
        el = soup.select_one('#bylineInfo')
        if el:
            d['brand'] = el.get_text(strip=True).replace('Visit the ', '').replace(' Store', '')

        d['specifications'] = self._extract_specs(soup)
        return d

    # ── Generic (Open Graph + Schema.org) ─────────────────────────
    def _generic(self, soup, url):
        d = {'source': 'generic', 'url': url, 'scrape_failed': False}
        d['name'] = self._og_value(soup, 'og:title') or (
            soup.title.string if soup.title else 'Unknown Product')
        d['description'] = (self._og_value(soup, 'og:description') or '')[:500]
        d['image'] = self._og_value(soup, 'og:image') or ''
        d['price'] = self._price_from_meta(soup)
        return d

    # ── timeout / block result ────────────────────────────────────
    @staticmethod
    def _timeout_result(url, domain):
        """Friendly fallback when site blocks or times out."""
        site = 'Flipkart' if 'flipkart' in domain else (
            'Amazon' if 'amazon' in domain else domain)
        return {
            'url': url,
            'name': '',
            'price': 0,
            'scrape_failed': True,
            'fail_reason': (
                f'{site} blocked the automated request. '
                f'This is normal — most e-commerce sites have anti-bot protection. '
                f'Please use the manual input form below to enter the product details.'
            ),
        }

    # ── helpers ───────────────────────────────────────────────────
    @staticmethod
    def _parse_price(el) -> float:
        if not el:
            return 0
        txt = el.get_text(strip=True)
        cleaned = re.sub(r'[^\d.]', '', txt)
        try:
            return float(cleaned)
        except ValueError:
            return 0

    @staticmethod
    def _og_value(soup, prop: str):
        tag = soup.find('meta', property=prop) or soup.find('meta', attrs={'name': prop})
        return tag.get('content', '').strip() if tag else None

    @classmethod
    def _price_from_meta(cls, soup) -> float:
        # Try product:price:amount meta tag
        tag = soup.find('meta', property='product:price:amount')
        if tag:
            try:
                return float(tag['content'])
            except (ValueError, TypeError, KeyError):
                pass
        # Try Schema.org JSON-LD
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                ld = json.loads(script.string)
                if isinstance(ld, dict):
                    offers = ld.get('offers', {})
                    if isinstance(offers, list):
                        offers = offers[0] if offers else {}
                    if isinstance(offers, dict) and 'price' in offers:
                        return float(offers['price'])
            except Exception:
                continue
        return 0

    @staticmethod
    def _extract_specs(soup) -> dict:
        specs = {}
        for row in (soup.select('div._14cfVK tr')
                    or soup.select('table._14cfVK tr')
                    or soup.select('table tr'))[:20]:
            cols = row.select('td')
            if len(cols) >= 2:
                specs[cols[0].get_text(strip=True)] = cols[1].get_text(strip=True)
        return specs
