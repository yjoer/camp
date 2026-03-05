export async function getIP() {
  const url = 'https://icanhazip.com';
  const response = await fetch(url);
  const data = await response.text();

  return data.trim();
}

export async function getLocation(ip: string) {
  const url = `http://ip-api.com/json/${ip}`;
  const response = await fetch(url);
  const data = await response.json() as { city: string; regionName: string; country: string };

  return `${data.city}, ${data.regionName}, ${data.country}`;
}
