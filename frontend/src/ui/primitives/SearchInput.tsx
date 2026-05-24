type SearchInputProps = {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string | undefined;
};

export const SearchInput = ({ value, onChange, placeholder }: SearchInputProps) => (
  <input
    type="search"
    value={value}
    onChange={(e) => onChange(e.target.value)}
    placeholder={placeholder}
    className="w-full rounded bg-black/60 text-fg-default px-3 py-2 outline-none"
  />
);
