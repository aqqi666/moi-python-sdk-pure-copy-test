"""Streaming response helpers for file downloads and SSE events."""

from __future__ import annotations

from typing import Iterator, Optional
import json
import os
import requests
import threading
import io

from .models import DataAnalysisStreamEvent


class FileStream:
    """Wraps a streaming HTTP response body."""

    def __init__(self, response: requests.Response):
        self._response = response
        self.body = response.raw
        self.headers = response.headers
        self.status_code = response.status_code

    def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
        """Iterate over the response body."""
        yield from self._response.iter_content(chunk_size)

    def read(self, size: int = -1) -> bytes:
        """Read raw bytes from the response."""
        return self._response.raw.read(size)

    def close(self) -> None:
        """Close the underlying HTTP response."""
        self._response.close()

    def write_to_file(self, file_path: str) -> int:
        """
        Write the stream content to a file at the specified path.
        
        The method creates the file and any necessary parent directories.
        It returns the number of bytes written.
        
        Args:
            file_path: Path to the output file
        
        Returns:
            Number of bytes written
        
        Example:
            stream = client.download_table_data({"id": 1})
            try:
                written = stream.write_to_file("/path/to/output.csv")
                print(f"Wrote {written} bytes to file")
            finally:
                stream.close()
        """
        if self._response is None or self.body is None:
            raise IOError("Stream is closed or invalid")
        
        # Create parent directories if they don't exist
        dir_path = os.path.dirname(file_path)
        if dir_path and dir_path != "":
            os.makedirs(dir_path, mode=0o755, exist_ok=True)
        
        # Write the stream content to the file
        written = 0
        with open(file_path, 'wb') as f:
            for chunk in self._response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
        
        return written

    def __enter__(self) -> "FileStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TimeoutIterator:
    """
    Wraps an iterator and provides timeout control that resets on each successful iteration.
    The timeout is applied to the interval between iterations, not the total iteration time.
    """
    
    def __init__(self, iterator: Iterator[str], timeout: float):
        """
        Initialize a TimeoutIterator.
        
        Args:
            iterator: The underlying iterator (e.g., iter_lines result)
            timeout: Timeout in seconds between iterations. If <= 0, no timeout is applied.
        """
        self._iterator = iterator
        self._timeout = timeout
        self._lock = threading.Lock()
    
    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        """
        Get the next item from the iterator with timeout.
        
        The timeout is reset each time an item is successfully retrieved.
        
        Returns:
            Next item from the iterator.
        
        Raises:
            TimeoutError: If no item is received within the timeout period.
            StopIteration: When the iterator is exhausted.
        """
        if self._timeout <= 0:
            # No timeout, iterate directly
            return next(self._iterator)
        
        # Serialize iterations to ensure timeout is properly reset
        with self._lock:
            # Use a queue to communicate between threads
            result_queue: list[Optional[tuple[Optional[str], Optional[Exception]]]] = []
            
            def iterate_operation():
                try:
                    item = next(self._iterator)
                    result_queue.append((item, None))
                except StopIteration:
                    result_queue.append((None, StopIteration()))
                except Exception as e:
                    result_queue.append((None, e))
            
            # Start iteration operation in a separate thread
            iterate_thread = threading.Thread(target=iterate_operation, daemon=True)
            iterate_thread.start()
            iterate_thread.join(timeout=self._timeout)
            
            if result_queue:
                item, error = result_queue[0]
                if error:
                    # Re-raise StopIteration or other exceptions
                    raise error
                # Iteration completed successfully - timeout is effectively reset for the next iteration
                return item
            else:
                # Timeout - no item received within the timeout period
                raise TimeoutError(f"read timeout: no data received within {self._timeout} seconds")


class DataAnalysisStream:
    """
    Wraps a streaming HTTP response for data analysis API.
    
    The stream returns Server-Sent Events (SSE) format. Use read_event to read
    individual events from the stream.
    
    The stream includes events such as:
    - init: Initialization event (first event) with request_id and session_title
      (step_type="init", data contains request_id and session_title)
    - classification: Question classification result
    - complete: Analysis complete
    - error: Error information
    
    Example:
        stream = client.analyze_data_stream({
            "question": "2024年收入下降的原因是什么？",
            "session_id": "session_123"
        })
        try:
            while True:
                event = stream.read_event()
                if event is None:  # EOF
                    break
                # Check for init event
                if event.step_type == "init":
                    init_data = event.get_init_event_data()
                    if init_data:
                        print(f"Request ID: {init_data.request_id}, Session Title: {init_data.session_title}")
                print(f"Event type: {event.type}")
        finally:
            stream.close()
    """

    def __init__(self, response: requests.Response, initial_buffer_size: int = 0, read_timeout: float = 30.0):
        """
        Initialize a DataAnalysisStream.
        
        Args:
            response: The HTTP response object
            initial_buffer_size: Initial buffer size for reading lines (in bytes).
                                Default is 0, which means use default chunk size (512 bytes).
                                The buffer will dynamically grow as needed to handle lines of
                                arbitrary length, so this option only sets the initial buffer
                                size for better performance. This is provided for API consistency
                                with Go SDK.
            read_timeout: Timeout in seconds between messages in streaming responses.
                         Default is 30.0 seconds. The timeout is reset each time data is
                         successfully read from the stream. If no data is received within this
                         timeout period, the read operation will fail. Set to 0 to disable
                         timeout (not recommended).
        """
        self._response = response
        self._read_timeout = read_timeout if read_timeout > 0 else 30.0  # Default 30 seconds
        self.headers = response.headers
        self.status_code = response.status_code
        self._scanner = None
        self._initial_buffer_size = initial_buffer_size
        self._wrapped_body = None

    @property
    def body(self):
        """Get the response body."""
        return self._response.raw

    def close(self) -> None:
        """Close the underlying HTTP response."""
        if self._response is not None:
            self._response.close()

    def read_event(self) -> Optional[DataAnalysisStreamEvent]:
        """
        Read the next SSE event from the stream.
        
        Returns:
            DataAnalysisStreamEvent or None when the stream is complete (EOF).
        
        Example:
            while True:
                event = stream.read_event()
                if event is None:
                    break
                # Process event
        """
        if self._scanner is None:
            # Initialize scanner to read line by line
            # Set chunk_size based on initial_buffer_size if specified
            # Default chunk_size is 512 bytes. iter_lines will dynamically grow to handle
            # lines of arbitrary length, so this only sets the initial chunk size.
            chunk_size = 512
            if self._initial_buffer_size > 0:
                # Use a reasonable chunk size based on initial buffer size
                # Default to 4KB if not specified, but allow custom sizes
                chunk_size = max(512, min(self._initial_buffer_size, 8192))  # Between 512B and 8KB
            else:
                # Default initial buffer size: 4KB (4096 bytes)
                chunk_size = 4096
            base_iterator = self._response.iter_lines(decode_unicode=True, chunk_size=chunk_size)
            # Wrap iterator with timeout if timeout is configured
            # The timeout is reset each time data is successfully read
            if self._read_timeout > 0:
                self._scanner = TimeoutIterator(iter(base_iterator), self._read_timeout)
            else:
                self._scanner = iter(base_iterator)

        event = DataAnalysisStreamEvent()
        data_lines = []
        event_type = None

        try:
            for line in self._scanner:
                line = line.strip()
                if line == "":
                    # Empty line indicates end of event
                    if len(data_lines) > 0:
                        # Parse the accumulated data
                        data_str = "\n".join(data_lines)
                        event.raw_data = data_str.encode('utf-8')
                        try:
                            parsed = json.loads(data_str)
                            if isinstance(parsed, dict):
                                # Extract all fields from parsed JSON
                                event.data = parsed
                                event.type = parsed.get("type") or event.type
                                event.source = parsed.get("source") or event.source
                                event.step_type = parsed.get("step_type") or event.step_type
                                event.step_name = parsed.get("step_name") or event.step_name
                        except json.JSONDecodeError:
                            # If JSON parsing fails, return raw data (event already has raw_data set)
                            # But still set event.type if event_type was specified
                            if event_type:
                                event.type = event_type
                            return event
                        if event_type:
                            event.type = event_type
                        return event
                    continue

                # Parse SSE format: "field: value"
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    data_lines.append(data)
                elif line.startswith("event: "):
                    event_type = line[7:]  # Remove "event: " prefix
                # Ignore other SSE fields (id, retry, etc.)
        
        except TimeoutError:
            # Timeout error - check if it contains "read timeout"
            raise

            # Handle last event if any
            if len(data_lines) > 0:
                data_str = "\n".join(data_lines)
                event.raw_data = data_str.encode('utf-8')
                try:
                    parsed = json.loads(data_str)
                    if isinstance(parsed, dict):
                        # Extract all fields from parsed JSON
                        event.data = parsed
                        event.type = parsed.get("type") or event.type
                        event.source = parsed.get("source") or event.source
                        event.step_type = parsed.get("step_type") or event.step_type
                        event.step_name = parsed.get("step_name") or event.step_name
                except json.JSONDecodeError:
                    # If JSON parsing fails, return raw data (event already has raw_data set)
                    # But still set event.type if event_type was specified
                    if event_type:
                        event.type = event_type
                    return event
                if event_type:
                    event.type = event_type
                return event

            # EOF
            return None

        except StopIteration:
            return None

    def __enter__(self) -> "DataAnalysisStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

