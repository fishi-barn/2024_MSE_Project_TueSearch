import re
import socket
from OpenSSL import SSL, crypto

class CertificateManager:
    def __init__(self, base_cert_path, adapt_cert_path):
        """
        Initialize the CertificateManager with paths to base and adaptive certificates.

        :param base_cert_path: Path to the base certificate file.
        :param adapt_cert_path: Path to the adaptive certificate file.
        """
        self.base_cert_path = base_cert_path
        self.adapt_cert_path = adapt_cert_path
        self.merged_cert_path = "./certs/merged_cert.pem"

    @staticmethod
    def load_custom_cert_store(filepath):
        """
        Load a custom certificate store from a given file path.

        :param filepath: Path to the certificate file.
        :return: SSL context with the loaded certificate.
        """
        store = SSL.Context(SSL.SSLv23_METHOD)
        store.load_verify_locations(filepath)
        return store

    @staticmethod
    def read_file(filepath):
        """
        Read the contents of a file.

        :param filepath: Path to the file.
        :return: Contents of the file.
        """
        with open(filepath, "r") as file:
            return file.read()

    @staticmethod
    def extract_domain(url):
        """
        Extract the domain from a given URL.

        :param url: The URL to extract the domain from.
        :return: Extracted domain.
        """
        pattern = r'(?<=://)(?:www\.)?(?P<domain>[^/]+)'
        match = re.search(pattern, url)
        if match:
            domain_parts = match.group('domain').split('.')
            return '.'.join(domain_parts[-2:])
        return None

    def get_pem_file(self, url, filepath):
        """
        Retrieve the PEM file from a given URL.

        :param url: The URL to retrieve the PEM file from.
        :param filepath: Path to the certificate file.
        :return: PEM file content.
        """
        domain = self.extract_domain(url)
        if not domain:
            raise ValueError("Invalid URL: Unable to extract domain.")

        dst = (domain, 443)
        ctx = self.load_custom_cert_store(filepath)

        try:
            sock = socket.create_connection(dst)
            ssl_sock = SSL.Connection(ctx, sock)
            ssl_sock.set_connect_state()
            ssl_sock.set_tlsext_host_name(domain.encode())
            ssl_sock.sendall(b'HEAD / HTTP/1.0\n\n')

            if ssl_sock.get_peer_certificate():
                ssl_sock.do_handshake()

            peer_cert_chain = ssl_sock.get_peer_cert_chain()
            pem_file = ''.join([crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8") for cert in peer_cert_chain])
            return pem_file
        except Exception as e:
            raise ConnectionError(f"Failed to retrieve PEM file: {e}")

    def combine_certs(self, url):
        """
        Combine base, adaptive, and website certificates into a merged certificate file.

        :param url: The URL to retrieve the website certificate from.
        """
        adapt_cert = self.read_file(self.adapt_cert_path)
        base_cert = self.read_file(self.base_cert_path)
        website_extra_certs = self.get_pem_file(url, self.base_cert_path) + "\n"

        if website_extra_certs not in adapt_cert:
            with open(self.adapt_cert_path, "a") as file:
                file.write(website_extra_certs + "\n")
            merged_certs = website_extra_certs + adapt_cert + base_cert
        else:
            merged_certs = adapt_cert + base_cert

        with open(self.merged_cert_path, "w") as file:
            file.write(merged_certs)
