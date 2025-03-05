import scrapy

class TuyaSpider(scrapy.Spider):
    name = "tuya"
    start_urls = ["https://www.tuya.com.co/como-pago-mi-tarjeta-o-credicompras","https://www.tuya.com.co/tarjetas-de-credito",
                  "https://www.tuya.com.co/credicompras","https://www.tuya.com.co/otras-soluciones-financieras","https://www.tuya.com.co/nuestra-compania",
                  "https://www.tuya.com.co/activacion-tarjeta"
                  ]

    def parse(self, response):
        # Extraer los títulos y textos de la página
        titles = response.css("h1, h2, h3::text").getall()
        paragraphs = response.css("p::text, p strong::text").getall()
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Extraer enlaces de pago
        links = response.css("a::attr(href)").getall()
        
        yield {
            "titles": titles,
            "paragraphs": paragraphs,
            "links": links,
        }



