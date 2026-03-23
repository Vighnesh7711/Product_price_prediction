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

        # Try multiple browser impersonations for better success rate
        impersonations = ["chrome110", "chrome120", "edge104", "safari15_5"]

        for i, impersonation in enumerate(impersonations):
            try:
                # Add realistic headers for the specific site
                headers = self._get_headers_for_domain(domain)

                # Impersonate a real browser TLS fingerprint with custom headers
                resp = requests.get(
                    url,
                    impersonate=impersonation,
                    timeout=30,  # Slightly longer timeout
                    allow_redirects=True,
                    headers=headers
                )

                # Check for specific anti-bot responses
                if resp.status_code == 403 and 'flipkart' in domain:
                    # Flipkart specific 403 handling
                    if 'blocked' in resp.text.lower() or 'captcha' in resp.text.lower():
                        raise Exception(f"HTTP Error {resp.status_code}: Bot detection")

                if resp.status_code >= 400:
                    if i < len(impersonations) - 1:  # Try next impersonation
                        continue
                    raise Exception(f"HTTP Error {resp.status_code}: {resp.reason}")

                # Check for empty or minimal content (potential blocking)
                if len(resp.text) < 1000:
                    if i < len(impersonations) - 1:  # Try next impersonation
                        continue
                    raise Exception("Minimal content received - likely blocked")

                soup = BeautifulSoup(resp.text, 'lxml')

                # Additional check for Flipkart: ensure we have some product-related content
                if 'flipkart' in domain:
                    if not soup.find(text=lambda text: text and ('price' in text.lower() or 'cart' in text.lower())):
                        if i < len(impersonations) - 1:  # Try next impersonation
                            continue

                # If we reach here, scraping seems successful
                break

            except Exception as e:
                err_str = str(e).lower()
                if ('timeout' in err_str or 'connection' in err_str or
                    'http error 529' in err_str or 'http error 403' in err_str or
                    'bot detection' in err_str):
                    if i < len(impersonations) - 1:  # Try next impersonation
                        continue
                    return self._timeout_result(url, domain)

                if i < len(impersonations) - 1:  # Try next impersonation
                    continue

                return {'error': str(e), 'url': url, 'name': '', 'price': 0,
                        'scrape_failed': True,
                        'fail_reason': f'Could not access the page: {e}'}

        if 'flipkart' in domain:
            return self._flipkart(soup, url)
        if 'amazon' in domain:
            return self._amazon(soup, url)
        return self._generic(soup, url)

    def _get_headers_for_domain(self, domain: str) -> dict:
        """Get realistic headers based on the target domain."""
        base_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        if 'flipkart' in domain:
            base_headers.update({
                'Referer': 'https://www.google.com/',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Cache-Control': 'max-age=0',
            })
        elif 'amazon' in domain:
            base_headers.update({
                'Referer': 'https://www.google.com/',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
            })

        return base_headers

    # ── Flipkart ──────────────────────────────────────────────────
    def _flipkart(self, soup, url):
        d = {'source': 'flipkart', 'url': url, 'scrape_failed': False}

        # Name — try multiple selectors (updated for 2026)
        name_selectors = [
            'span.VU-ZEz',           # Original selector
            'span.B_NuCI',           # Original selector
            'h1 span',               # Generic h1 span
            'h1',                    # Just h1
            '[data-hook="product-title"]',  # Data hook selector
            '.product-title',         # Class-based
            'h1._35KyD6',            # Updated Flipkart selector
            'h1.x-product-title-label', # Another possible selector
            '[data-testid="product-title"]'  # Test id selector
        ]

        for sel in name_selectors:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                d['name'] = el.get_text(strip=True)
                break
        else:
            d['name'] = self._og_value(soup, 'og:title') or 'Unknown Product'

        # Price — updated selectors for 2026
        price_selectors = [
            'div.Nx9bqj.CxhGGd',      # Original selector
            'div._30jeq3._16Jk6d',    # Original selector
            'div._30jeq3',            # Original selector
            'div._1_WHN1',           # Updated selector
            'span._30jeq3',          # Span variant
            '[data-hook="selling-price"]', # Data hook
            '.selling-price',        # Class-based
            'div.CEmiEU div.Nx9bqj', # More specific
            '[data-testid="price-mp-label"]', # Test id
            '.current-price'         # Generic current price
        ]

        for sel in price_selectors:
            el = soup.select_one(sel)
            if el:
                price = self._parse_price(el)
                if price > 0:
                    d['price'] = price
                    break

        if not d.get('price'):
            d['price'] = self._price_from_meta(soup)

        # Original price — updated selectors
        original_price_selectors = [
            'div.yRaY8j.A6\\+E6v',    # Original selector (escaped +)
            'div._3I9_wc._2p6lqe',    # Original selector
            'div.yRaY8j',            # Original selector
            'div._3auQ3N._2GcJzG',   # Updated selector
            'span.yRaY8j',           # Span variant
            '[data-hook="list-price"]', # Data hook
            '.list-price',           # Class-based
            'div._3I9_wc',           # Simplified
            '[data-testid="list-price-mp-label"]' # Test id
        ]

        for sel in original_price_selectors:
            try:
                el = soup.select_one(sel)
                if el:
                    orig_price = self._parse_price(el)
                    if orig_price > 0:
                        d['original_price'] = orig_price
                        break
            except:
                continue

        # Rating — updated selectors
        rating_selectors = [
            'div.XQDdHH',           # Original selector
            'div._3LWZlK',          # Original selector
            'div._3LWZlK.UvPGHe',  # More specific
            'span._1lRcqv',        # Alternative selector
            '[data-hook="rating-label"]', # Data hook
            '.rating-label',       # Class-based
            'div.Wphh3N',         # Updated selector
            '[data-testid="stars-rating-label"]', # Test id
            'span[data-testid="average-rating"]'  # Test id variant
        ]

        for sel in rating_selectors:
            try:
                el = soup.select_one(sel)
                if el:
                    rating_text = el.get_text(strip=True)
                    try:
                        # Extract first number from rating text
                        import re
                        match = re.search(r'(\d+\.?\d*)', rating_text)
                        if match:
                            d['rating'] = float(match.group(1))
                            break
                    except (ValueError, IndexError):
                        continue
            except:
                continue

        # Description — updated selectors
        desc_selectors = [
            'div._1mXcCf',           # Original selector
            'div._1AN87F',           # Original selector
            'div.RmoJUa',            # Original selector
            'div._4cdFB2',          # Updated selector
            'div.UHZEA8',           # Alternative
            '[data-hook="product-description"]', # Data hook
            '.product-description',  # Class-based
            'div[data-testid="product-description"]', # Test id
            'div.product-details'    # Generic
        ]

        for sel in desc_selectors:
            try:
                el = soup.select_one(sel)
                if el:
                    desc_text = el.get_text(strip=True)
                    if desc_text:
                        d['description'] = desc_text[:500]
                        break
            except:
                continue

        if not d.get('description'):
            d['description'] = self._og_value(soup, 'og:description') or ''

        # Image — updated selectors
        image_selectors = [
            'img._396cs4._2amPTt._3qGmMb', # Original selector
            'img._396cs4',                # Original selector
            'img.DByuf4',                # Original selector
            'img._2r_T1I',               # Updated selector
            'img[data-hook="product-image"]', # Data hook
            '.product-image img',        # Class-based
            'img[data-testid="product-image"]', # Test id
            'img._53J4C-._6LmNvR',      # Alternative
            'img[alt*="product"]',       # Alt text contains product
            '.product-gallery img'      # Generic gallery image
        ]

        for sel in image_selectors:
            try:
                el = soup.select_one(sel)
                if el and el.get('src'):
                    d['image'] = el.get('src', '')
                    break
            except:
                continue

        if not d.get('image'):
            d['image'] = self._og_value(soup, 'og:image') or ''

        # Category breadcrumbs — updated selectors
        breadcrumb_selectors = [
            'a._2whKao',               # Original selector
            'div._1MR4o5 a',           # Original selector with div
            'nav[data-testid="breadcrumb"] a', # Testid breadcrumb
            '.breadcrumb a',           # Generic breadcrumb
            'div._3GIHBu a',          # Alternative selector
            'div[data-hook="breadcrumb"] a', # Data hook
            '.navigation-breadcrumb a' # Generic navigation
        ]

        for breadcrumb_sel in breadcrumb_selectors:
            try:
                crumbs = soup.select(breadcrumb_sel)
                if crumbs:
                    crumb_texts = [c.get_text(strip=True) for c in crumbs if c.get_text(strip=True)]
                    if crumb_texts:
                        d['category_path'] = ' >> '.join(crumb_texts)
                        break
            except:
                continue

        # Specifications
        d['specifications'] = self._extract_specs(soup)

        # Add debug info for troubleshooting
        if not d.get('name') or d['name'] == 'Unknown Product':
            d['debug_name_issue'] = True
        if not d.get('price') or d['price'] == 0:
            d['debug_price_issue'] = True

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

        # Remove common currency symbols and text
        txt = re.sub(r'[₹$€£,\s]', '', txt)
        txt = re.sub(r'[^\d\.]', '', txt)

        # Handle different price formats
        if '.' in txt:
            parts = txt.split('.')
            if len(parts) == 2 and len(parts[1]) <= 2:  # Decimal format
                try:
                    return float(txt)
                except ValueError:
                    return 0
            else:  # Thousands separator format
                txt = txt.replace('.', '')

        try:
            return float(txt)
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
