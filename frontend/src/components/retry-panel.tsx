export function RetryPanel({ message }: { message: string }) {
  return (
    <div className="card mt-8 p-6 text-sm" role="alert">
      <p className="font-semibold text-red-700">{message}</p>
      <button
        type="button"
        onClick={() => window.location.reload()}
        className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-xs font-bold text-white"
      >
        Retry page
      </button>
    </div>
  );
}
