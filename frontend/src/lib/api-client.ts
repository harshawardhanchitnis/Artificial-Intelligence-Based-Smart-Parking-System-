export const apiBase =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

type ErrorPayload = {
  detail?: string;
  error?: { message?: string; request_id?: string };
};

export class ApiClientError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
    readonly requestId: string | null,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

export type ApiFetchOptions = RequestInit & {
  timeoutMs?: number;
  acceptedStatuses?: number[];
};

export async function apiFetch<T>(path: string, options: ApiFetchOptions = {}): Promise<T> {
  const { timeoutMs = 12_000, acceptedStatuses = [], headers, ...requestOptions } = options;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(apiUrl(path), {
      ...requestOptions,
      cache: requestOptions.cache ?? "no-store",
      headers: { Accept: "application/json", ...headers },
      signal: controller.signal,
    });
    const requestId = response.headers.get("x-request-id");
    let payload: T | ErrorPayload | null = null;
    try {
      payload = (await response.json()) as T | ErrorPayload;
    } catch {
      if (response.ok) throw new ApiClientError("The local API returned invalid JSON", response.status, requestId);
    }
    if (!response.ok && !acceptedStatuses.includes(response.status)) {
      const failure = payload as ErrorPayload | null;
      const message = failure?.error?.message ?? failure?.detail ?? `Local API request failed (${response.status})`;
      throw new ApiClientError(message, response.status, requestId ?? failure?.error?.request_id ?? null);
    }
    return payload as T;
  } catch (reason) {
    if (reason instanceof ApiClientError) throw reason;
    if (reason instanceof DOMException && reason.name === "AbortError") {
      throw new ApiClientError("The local API request timed out", null, null);
    }
    throw new ApiClientError("The local API is unavailable", null, null);
  } finally {
    window.clearTimeout(timeout);
  }
}

export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;

  const normalized =
    path === "/api/v1"
      ? ""
      : path.startsWith("/api/v1/")
        ? path.slice("/api/v1".length)
        : path;

  return `${apiBase}${normalized.startsWith("/") ? normalized : `/${normalized}`}`;
}

export function apiErrorMessage(reason: unknown): string {
  if (!(reason instanceof ApiClientError)) return "The local API request failed.";
  return reason.requestId ? `${reason.message} (request ${reason.requestId})` : reason.message;
}
